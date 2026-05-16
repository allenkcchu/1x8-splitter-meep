#!/bin/bash
source /home/allen/anaconda3/etc/profile.d/conda.sh
conda activate pmp
cd /home/allen/meep_notebooks

python /mnt/c/Users/Hemera/Projects/meep_1x8_progress/scripts/eval_param.py --variant L40W14 \
  >> /mnt/c/Users/Hemera/Projects/meep_1x8_progress/stage1_L40W14/eval.log 2>&1

echo "DONE $(date)" >> /mnt/c/Users/Hemera/Projects/meep_1x8_progress/stage1_L40W14/eval.log
