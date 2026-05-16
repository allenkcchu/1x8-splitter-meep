#!/bin/bash
# Stage 1: Gaussian filter adjoint optimization, 100 iterations
# Run from WSL: bash /mnt/c/Users/Hemera/Projects/meep_1x8_progress/run_stage1.sh

source /home/allen/anaconda3/etc/profile.d/conda.sh
conda activate pmp
cd /home/allen/meep_notebooks

cp /mnt/c/Users/Hemera/Projects/meep_1x8_progress/meep_1x8_splitter.ipynb .
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=43200 \
  --ExecutePreprocessor.kernel_name=pmp \
  meep_1x8_splitter.ipynb \
  >> /mnt/c/Users/Hemera/Projects/meep_1x8_progress/nbconvert.log 2>&1

cp meep_1x8_splitter.ipynb /mnt/c/Users/Hemera/Projects/meep_1x8_progress/
echo "DONE $(date)" >> /mnt/c/Users/Hemera/Projects/meep_1x8_progress/nbconvert.log
