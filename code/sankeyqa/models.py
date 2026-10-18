"""Model adapters. Every adapter answers one question at a time and returns the raw reply and the time it took.

Image readers receive the PNG of the diagram, table readers receive the flow table written as text. All models are
small open models that run on a laptop CPU, with greedy decoding so that runs are repeatable.
"""
from __future__ import annotations

import json
import time
from typing import Any

import torch
from PIL import Image

torch.set_num_threads(max(1, torch.get_num_threads()))

INSTRUCTION = ("Answer the question about the career map with a short answer only, a name, a number, yes or no, "
               "or increase, decrease or stay the same.")

MODELS = {
    "smolvlm-256m": "HuggingFaceTB/SmolVLM-256M-Instruct",
    "smolvlm-500m": "HuggingFaceTB/SmolVLM-500M-Instruct",
    "blip-vqa": "Salesforce/blip-vqa-base",
    "matcha-chartqa": "google/matcha-chartqa",
    "deplot": "google/deplot",
    "qwen2.5-0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen2.5-1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
}
REVISIONS: dict[str, str] = {}   # filled from the Hugging Face API at run time and stored with the results


def _load_rgb(path: str) -> Image.Image:
    im = Image.open(path)
    if im.mode in ("RGBA", "LA", "P"):
        # flatten on black, which is what a reader that ignores the alpha channel effectively sees
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, (0, 0, 0, 255))
        im = Image.alpha_composite(bg, im)
    return im.convert("RGB")


class SmolVLM:
    def __init__(self, name: str):
        from transformers import AutoProcessor, AutoModelForImageTextToText
        self.repo = MODELS[name]
        self.proc = AutoProcessor.from_pretrained(self.repo)
        self.model = AutoModelForImageTextToText.from_pretrained(self.repo, dtype=torch.float32).eval()

    @torch.inference_mode()
    def ask(self, image_path: str, question: str) -> tuple[str, float]:
        t = time.time()
        msgs = [{"role": "user", "content": [{"type": "image"},
                                              {"type": "text", "text": f"{INSTRUCTION}\n{question}"}]}]
        prompt = self.proc.apply_chat_template(msgs, add_generation_prompt=True)
        inp = self.proc(text=prompt, images=[_load_rgb(image_path)], return_tensors="pt")
        out = self.model.generate(**inp, max_new_tokens=24, do_sample=False)
        txt = self.proc.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        return txt.strip(), time.time() - t


class BlipVQA:
    def __init__(self, name: str = "blip-vqa", path: str | None = None):
        from transformers import BlipProcessor, BlipForQuestionAnswering
        self.repo = path or MODELS["blip-vqa"]
        self.proc = BlipProcessor.from_pretrained(MODELS["blip-vqa"])
        self.model = BlipForQuestionAnswering.from_pretrained(self.repo, dtype=torch.float32).eval()

    @torch.inference_mode()
    def ask(self, image_path: str, question: str) -> tuple[str, float]:
        t = time.time()
        inp = self.proc(images=_load_rgb(image_path), text=question, return_tensors="pt")
        out = self.model.generate(**inp, max_new_tokens=12)
        return self.proc.decode(out[0], skip_special_tokens=True).strip(), time.time() - t


class Pix2Struct:
    """MatCha fine-tuned on ChartQA answers the question directly. DePlot converts the chart to a table, which a
    table reader then uses (see TwoStage)."""
    def __init__(self, name: str):
        from transformers import Pix2StructProcessor, Pix2StructForConditionalGeneration
        self.repo = MODELS[name]
        self.proc = Pix2StructProcessor.from_pretrained(self.repo)
        self.model = Pix2StructForConditionalGeneration.from_pretrained(self.repo, dtype=torch.float32).eval()
        self.max_new = 24 if name == "matcha-chartqa" else 512

    @torch.inference_mode()
    def ask(self, image_path: str, question: str) -> tuple[str, float]:
        t = time.time()
        inp = self.proc(images=_load_rgb(image_path), text=question, return_tensors="pt", max_patches=1024)
        out = self.model.generate(**inp, max_new_tokens=self.max_new)
        return self.proc.decode(out[0], skip_special_tokens=True).strip(), time.time() - t


def table_text(t: dict[str, Any]) -> str:
    """The flow table as compact text, as an application would pass it to a language model."""
    cfg = t["config"]
    lines = [f"Career map ({'roles' if cfg['map_type'] == 'role' else 'industries'}), {t['n_profiles']} profiles "
             f"in total, stages {', '.join(t['stages'])}."]
    for s, st in enumerate(t["nodes"]):
        lines.append(f"Stage {t['stages'][s]}: " + ", ".join(f"{d['label']} {d['count']}" for d in st))
    for s in range(3):
        ls = [l for l in t["links"] if l["stage"] == s]
        lines.append(f"Flows {t['stages'][s]} to {t['stages'][s + 1]}: "
                     + ", ".join(f"{l['source']} to {l['target']} {l['count']}" for l in ls))
    return "\n".join(lines)


class TableLLM:
    """A small instruction-tuned language model that reads the flow table as text instead of the picture."""
    def __init__(self, name: str):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.repo = MODELS[name]
        self.tok = AutoTokenizer.from_pretrained(self.repo)
        dtype = torch.bfloat16 if "1.5b" in name else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(self.repo, dtype=dtype).eval()

    @torch.inference_mode()
    def ask_text(self, context: str, question: str) -> tuple[str, float]:
        t = time.time()
        msgs = [{"role": "system", "content": INSTRUCTION},
                {"role": "user", "content": f"{context}\n\nQuestion: {question}\nShort answer:"}]
        ids = self.tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)
        out = self.model.generate(**ids, max_new_tokens=24, do_sample=False)
        txt = self.tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)
        return txt.strip(), time.time() - t


def load(name: str):
    if name.startswith("smolvlm"):
        return SmolVLM(name)
    if name == "blip-vqa":
        return BlipVQA()
    if name in ("matcha-chartqa", "deplot"):
        return Pix2Struct(name)
    if name.startswith("qwen2.5"):
        return TableLLM(name)
    raise KeyError(name)


def dumps(x: Any) -> str:
    return json.dumps(x, ensure_ascii=False)
