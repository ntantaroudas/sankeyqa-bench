#!/bin/bash
# Resumable queue of every remaining run. Each step skips what is already done, so the queue can be restarted.
cd "$(dirname "$0")/.."
LOG=../results/logs
F="warn|Loading weights|max_length|generation_config"
ev() { python3 scripts/run_eval.py "$@" 2>&1 | grep -v -i -E "$F" >> $LOG/eval.log; }
ev --reader matcha-chartqa --set main
ev --reader smolvlm-256m --set main
ev --reader table:qwen2.5-0.5b --set main
ev --reader smolvlm-500m --set main
FT=../results/finetune_runs
run_ft() { name=$1; shift; [ -f $FT/$name/train_info.json ] || python3 scripts/finetune_blip.py "$@" --out $FT/$name 2>&1 | grep -v -i -E "$F" >> $LOG/finetune.log; ev --reader blip-ft:$FT/$name --set main --outset finetune; }
run_ft kmeans20 --selection kmeans --n-images 20 --seed 0
run_ft random20_s1 --selection random --n-images 20 --seed 1
run_ft random100 --selection random --n-images 100 --seed 1
run_ft random400 --selection random --n-images 400 --seed 1
run_ft random20_s2 --selection random --n-images 20 --seed 2
run_ft random20_s3 --selection random --n-images 20 --seed 3
ev --reader blip-vqa --set robust
ev --reader smolvlm-256m --set robust
echo QUEUE_ALL_DONE
