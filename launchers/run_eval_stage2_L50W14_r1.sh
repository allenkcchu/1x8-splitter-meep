#!/bin/bash
source /home/allen/anaconda3/etc/profile.d/conda.sh
conda activate pmp
cd /home/allen/meep_notebooks

python /mnt/c/Users/Hemera/Projects/meep_1x8_progress/scripts/eval_stage2.py \
  --variant L50W14 \
  --base /mnt/c/Users/Hemera/Projects/meep_1x8_progress/meep \
  --out  /mnt/c/Users/Hemera/Projects/meep_1x8_progress/meep/stage2_L50W14_r1 \
  >> /mnt/c/Users/Hemera/Projects/meep_1x8_progress/meep/stage2_L50W14_r1/eval.log 2>&1

echo "DONE $(date)" >> /mnt/c/Users/Hemera/Projects/meep_1x8_progress/meep/stage2_L50W14_r1/eval.log
