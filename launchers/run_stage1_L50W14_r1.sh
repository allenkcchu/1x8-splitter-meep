#!/bin/bash
source /home/allen/anaconda3/etc/profile.d/conda.sh
conda activate pmp
cd /home/allen/meep_notebooks

mkdir -p /mnt/c/Users/Hemera/Projects/meep_1x8_progress/meep/stage1_L50W14_r1

python /mnt/c/Users/Hemera/Projects/meep_1x8_progress/scripts/stage1_param.py \
  --mmi_L 50 --mmi_W 14 --n_iters 30 \
  --warmstart /mnt/c/Users/Hemera/Projects/meep_1x8_progress/meep/stage2_L50W14/x_latest.npy \
  --warmgeo   /mnt/c/Users/Hemera/Projects/meep_1x8_progress/meep/stage1_L50W14/geo_final.json \
  --out       /mnt/c/Users/Hemera/Projects/meep_1x8_progress/meep/stage1_L50W14_r1 \
  >> /mnt/c/Users/Hemera/Projects/meep_1x8_progress/meep/stage1_L50W14_r1/run.log 2>&1

echo "DONE $(date)" >> /mnt/c/Users/Hemera/Projects/meep_1x8_progress/meep/stage1_L50W14_r1/run.log
