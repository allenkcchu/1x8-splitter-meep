#!/bin/bash
# Stage 1 parametric: mmi_L=50um, mmi_W=14um
source /home/allen/anaconda3/etc/profile.d/conda.sh
conda activate pmp
mkdir -p /mnt/c/Users/Hemera/Projects/meep_1x8_progress/stage1_L50W14
python /mnt/c/Users/Hemera/Projects/meep_1x8_progress/scripts/stage1_param.py \
  --mmi_L 50 --mmi_W 14 \
  >> /mnt/c/Users/Hemera/Projects/meep_1x8_progress/stage1_L50W14/run.log 2>&1
echo "DONE $(date)" >> /mnt/c/Users/Hemera/Projects/meep_1x8_progress/stage1_L50W14/run.log
