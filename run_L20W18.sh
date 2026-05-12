#!/bin/bash
source /home/allen/anaconda3/etc/profile.d/conda.sh
conda activate pmp
cd /home/allen/meep_notebooks

python /mnt/c/Users/Hemera/Projects/meep_1x8_progress/scripts/stage1_param.py --mmi_L 20 --mmi_W 18 \
  >> /mnt/c/Users/Hemera/Projects/meep_1x8_progress/stage1_L20W18/run.log 2>&1

echo "DONE $(date)" >> /mnt/c/Users/Hemera/Projects/meep_1x8_progress/stage1_L20W18/run.log
