#!/bin/bash
# Transmission evaluation for Stage 1 v2
# Run from WSL: bash /mnt/c/Users/Hemera/Projects/meep_1x8_progress/run_eval_v2.sh

source /home/allen/anaconda3/etc/profile.d/conda.sh
conda activate pmp
cd /home/allen/meep_notebooks

python /mnt/c/Users/Hemera/Projects/meep_1x8_progress/scripts/eval_v2.py \
  >> /mnt/c/Users/Hemera/Projects/meep_1x8_progress/stage1_v2/eval_v2.log 2>&1

echo "DONE $(date)" >> /mnt/c/Users/Hemera/Projects/meep_1x8_progress/stage1_v2/eval_v2.log
