#!/bin/bash
# Build the GitHub tree (release/sankeyqa-bench) and the Zenodo zip, without model weights or caches.
set -e
cd "$(dirname "$0")/../.."
OUT=release/sankeyqa-bench
rm -rf "$OUT" && mkdir -p "$OUT"
cp README.md LICENSE LICENSE-DATA.md CITATION.cff .zenodo.json requirements.txt "$OUT"/
cp -r code "$OUT"/ && find "$OUT/code" -name __pycache__ -o -name .pytest_cache | xargs rm -rf
cp -r data figures extended_data "$OUT"/
mkdir -p "$OUT/results/finetune_runs"
cp -r results/raw results/tables results/logs results/key_numbers.json results/model_versions.json "$OUT/results/"
for d in results/finetune_runs/*/; do n=$(basename "$d"); mkdir -p "$OUT/results/finetune_runs/$n"; cp "$d/train_info.json" "$OUT/results/finetune_runs/$n/"; done
mkdir -p "$OUT/paper/ore_template" "$OUT/preprints"
cp paper/*.py paper/*.md paper/references.yaml paper/reference_check.* "$OUT/paper/" 2>/dev/null || true
rm -f "$OUT/paper/manuscript_ore.expanded.md" "$OUT/paper/sections_methods.md" "$OUT/paper/sections_methods2.md"
cp paper/ore_template/fill_template.py paper/ore_template/ORE_Article_Template.docx "$OUT/paper/ore_template/"
cp preprints/main.src.tex preprints/build_preprints.py preprints/robustness_short.tex "$OUT/preprints/"
cp -r preprints/Definitions "$OUT/preprints/"
cat > .gitignore.tmp <<'G'
__pycache__/
*.pyc
results/finetune_runs/*/model.safetensors
results/blip_train_features.pt
*.aux
*.log
*.out
G
mv .gitignore.tmp "$OUT/.gitignore"
(cd release && rm -f sankeyqa-bench-v1.1.0.zip && zip -qr sankeyqa-bench-v1.1.0.zip sankeyqa-bench)
du -sh "$OUT" release/sankeyqa-bench-v1.1.0.zip
