#!/bin/bash
source /home/allen/anaconda3/etc/profile.d/conda.sh
conda activate pmp
cd /home/allen/meep_notebooks

mkdir -p /mnt/c/Users/Hemera/Projects/meep_1x8_progress/stage2_L20W18

python /mnt/c/Users/Hemera/Projects/meep_1x8_progress/scripts/stage2_param.py \
  --variant L20W18 --n_iters 50 --lr 0.01 --alpha 0.12 \
  >> /mnt/c/Users/Hemera/Projects/meep_1x8_progress/stage2_L20W18/run.log 2>&1

echo "DONE $(date)" >> /mnt/c/Users/Hemera/Projects/meep_1x8_progress/stage2_L20W18/run.log
