#!/usr/bin/env python3
"""Fine-tune BLIP (visual question answering, base size) on SankeyQA training diagrams, on a CPU.

    python scripts/finetune_blip.py --selection kmeans --n-images 20 --steps 600 --out runs/blip_kmeans20
    python scripts/finetune_blip.py --selection random --n-images 20 --seed 1 --steps 600 --out runs/blip_random20_s1

In the XR-CareerAssist project the team fine-tuned BLIP on career Sankey diagrams and found that a small set of 20
diagrams chosen by k-means clustering generalised better than 1,280 similar diagrams. This script repeats that
comparison under a fixed training budget. Every run makes the same number of updates on the same batch size, so the
runs differ only in which training diagrams they see.

  * kmeans   the diagram nearest to each centre of a k-means clustering (k = n_images) of the image features
  * random   n_images diagrams drawn at random from the training split

The image encoder is frozen and its outputs are computed once per diagram, which makes CPU training feasible. The
question encoder and the answer decoder are trained with AdamW. All questions of the selected diagrams are used.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from sankeyqa.models import _load_rgb  # noqa: E402

REPO = "Salesforce/blip-vqa-base"


def image_features(model, proc, paths, cache):
    feats = {}
    for p in paths:
        if p in cache:
            feats[p] = cache[p]
            continue
        with torch.inference_mode():
            pv = proc(images=_load_rgb(p), return_tensors="pt")["pixel_values"]
            emb = model.vision_model(pixel_values=pv)[0][0]
        cache[p] = emb.to(torch.float16)
        feats[p] = cache[p]
    return feats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection", choices=["kmeans", "random"], required=True)
    ap.add_argument("--n-images", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--data", default=os.path.join(HERE, "..", "..", "data"))
    ap.add_argument("--features", default=os.path.join(HERE, "..", "..", "results", "blip_train_features.pt"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    from transformers import BlipProcessor, BlipForQuestionAnswering
    proc = BlipProcessor.from_pretrained(REPO)
    model = BlipForQuestionAnswering.from_pretrained(REPO, dtype=torch.float32)
    sp = json.load(open(os.path.join(a.data, "splits.json")))
    train_ids = sp["train"]
    paths = {c: os.path.join(a.data, "images", "default", f"{c}.png") for c in train_ids}

    # image features of every training diagram, computed once and shared by all runs
    t0 = time.time()
    cache = torch.load(a.features) if os.path.exists(a.features) else {}
    n_before = len(cache)
    model.vision_model.eval()
    feats = image_features(model, proc, [paths[c] for c in train_ids], cache)
    if len(cache) > n_before:
        os.makedirs(os.path.dirname(a.features), exist_ok=True)
        torch.save(cache, a.features)
    print(f"features of {len(feats)} training diagrams ready in {time.time() - t0:.0f}s", flush=True)

    # selection
    if a.selection == "kmeans":
        from sklearn.cluster import KMeans
        X = np.stack([feats[paths[c]].float().mean(0).numpy() for c in train_ids])
        X = X / np.linalg.norm(X, axis=1, keepdims=True)
        km = KMeans(n_clusters=a.n_images, n_init=10, random_state=a.seed).fit(X)
        chosen = []
        for k in range(a.n_images):
            idx = np.where(km.labels_ == k)[0]
            d = np.linalg.norm(X[idx] - km.cluster_centers_[k], axis=1)
            chosen.append(train_ids[int(idx[np.argmin(d)])])
    else:
        chosen = [train_ids[i] for i in rng.choice(len(train_ids), size=a.n_images, replace=False)]
    qs = [json.loads(l) for l in open(os.path.join(a.data, "questions.jsonl"), encoding="utf-8")]
    keep = set(chosen)
    train = [q for q in qs if q["cohort"] in keep]
    print(f"{a.selection}: {len(chosen)} diagrams, {len(train)} questions", flush=True)

    # training: frozen image encoder, trainable question encoder and answer decoder
    for p in model.vision_model.parameters():
        p.requires_grad = False
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 30) * max(0.0, 1 - s / a.steps))
    model.train()
    model.vision_model.eval()
    losses = []
    order = []
    t0 = time.time()
    for step in range(a.steps):
        if len(order) < a.batch:
            order += list(rng.permutation(len(train)))
        batch = [train[i] for i in order[:a.batch]]
        order = order[a.batch:]
        img = torch.stack([feats[paths[q["cohort"]]].float() for q in batch])
        enc = proc.tokenizer([q["text"] for q in batch], padding=True, truncation=True, max_length=64,
                             return_tensors="pt")
        lab = proc.tokenizer([str(q["answer"]) for q in batch], padding=True, return_tensors="pt")
        q_emb = model.text_encoder(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                                   encoder_hidden_states=img, return_dict=False)[0]
        labels = lab["input_ids"].masked_fill(lab["input_ids"] == proc.tokenizer.pad_token_id, -100)
        dec_in = lab["input_ids"].clone()
        dec_in[:, 0] = model.decoder_start_token_id if hasattr(model, "decoder_start_token_id") else model.config.text_config.bos_token_id
        out = model.text_decoder(input_ids=dec_in, attention_mask=lab["attention_mask"], encoder_hidden_states=q_emb,
                                 labels=labels, return_dict=True, reduction="mean")
        loss = out.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        opt.zero_grad()
        losses.append(float(loss))
        if (step + 1) % 50 == 0:
            print(f"  step {step + 1}/{a.steps} loss {np.mean(losses[-50:]):.3f} {time.time() - t0:.0f}s", flush=True)
    os.makedirs(a.out, exist_ok=True)
    model.eval()
    model.save_pretrained(a.out)
    json.dump({"selection": a.selection, "n_images": a.n_images, "seed": a.seed, "steps": a.steps, "batch": a.batch,
               "lr": a.lr, "n_questions": len(train), "epochs": a.steps * a.batch / len(train), "chosen": chosen,
               "loss_first50": float(np.mean(losses[:50])), "loss_last50": float(np.mean(losses[-50:])),
               "train_seconds": time.time() - t0, "losses": losses}, open(os.path.join(a.out, "train_info.json"), "w"),
              indent=1)
    print(f"saved {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
