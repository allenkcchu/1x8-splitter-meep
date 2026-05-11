#!/bin/bash
# Transmission evaluation — run after Stage 1 or Stage 2
# From WSL: bash /mnt/c/Users/Hemera/Projects/meep_1x8_progress/run_eval.sh

source /home/allen/anaconda3/etc/profile.d/conda.sh
conda activate pmp
cd /home/allen/meep_notebooks

cp /mnt/c/Users/Hemera/Projects/meep_1x8_progress/meep_1x8_eval.ipynb .
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=3600 \
  --ExecutePreprocessor.kernel_name=pmp \
  meep_1x8_eval.ipynb \
  >> /mnt/c/Users/Hemera/Projects/meep_1x8_progress/eval.log 2>&1

cp meep_1x8_eval.ipynb /mnt/c/Users/Hemera/Projects/meep_1x8_progress/
echo "DONE $(date)" >> /mnt/c/Users/Hemera/Projects/meep_1x8_progress/eval.log
