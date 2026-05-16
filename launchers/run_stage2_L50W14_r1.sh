#!/bin/bash
source /home/allen/anaconda3/etc/profile.d/conda.sh
conda activate pmp
cd /home/allen/meep_notebooks

mkdir -p /mnt/c/Users/Hemera/Projects/meep_1x8_progress/meep/stage2_L50W14_r1

python /mnt/c/Users/Hemera/Projects/meep_1x8_progress/scripts/stage2_param.py \
  --variant L50W14 \
  --base /mnt/c/Users/Hemera/Projects/meep_1x8_progress/meep \
  --resume \
  --out /mnt/c/Users/Hemera/Projects/meep_1x8_progress/meep/stage2_L50W14_r1 \
  --start_iter 0 --n_iters 30 --lr 0.01 --alpha 0.12 \
  >> /mnt/c/Users/Hemera/Projects/meep_1x8_progress/meep/stage2_L50W14_r1/run.log 2>&1

echo "DONE $(date)" >> /mnt/c/Users/Hemera/Projects/meep_1x8_progress/meep/stage2_L50W14_r1/run.log
