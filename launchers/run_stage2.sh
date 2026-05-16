#!/bin/bash
# Stage 2: Conic filter refinement, warm start from Stage 1
# Run from WSL: bash /mnt/c/Users/Hemera/Projects/meep_1x8_progress/run_stage2.sh

source /home/allen/anaconda3/etc/profile.d/conda.sh
conda activate pmp
cd /home/allen/meep_notebooks

mkdir -p /mnt/c/Users/Hemera/Projects/meep_1x8_progress/stage2

cp /mnt/c/Users/Hemera/Projects/meep_1x8_progress/meep_1x8_stage2.ipynb .
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=43200 \
  --ExecutePreprocessor.kernel_name=pmp \
  meep_1x8_stage2.ipynb \
  >> /mnt/c/Users/Hemera/Projects/meep_1x8_progress/stage2/nbconvert.log 2>&1

cp meep_1x8_stage2.ipynb /mnt/c/Users/Hemera/Projects/meep_1x8_progress/
echo "DONE $(date)" >> /mnt/c/Users/Hemera/Projects/meep_1x8_progress/stage2/nbconvert.log
