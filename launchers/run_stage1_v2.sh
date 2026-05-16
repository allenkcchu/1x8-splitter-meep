#!/bin/bash
# Stage 1 v2: Topology + Geometry joint optimization
# Uniformity penalty + port_pitch / wg_w_in / wg_w_out as learnable params
# Run from WSL: bash /mnt/c/Users/Hemera/Projects/meep_1x8_progress/run_stage1_v2.sh

source /home/allen/anaconda3/etc/profile.d/conda.sh
conda activate pmp
cd /home/allen/meep_notebooks

mkdir -p /mnt/c/Users/Hemera/Projects/meep_1x8_progress/stage1_v2

cp /mnt/c/Users/Hemera/Projects/meep_1x8_progress/meep_1x8_splitter_v2.ipynb .
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=86400 \
  --ExecutePreprocessor.kernel_name=pmp \
  meep_1x8_splitter_v2.ipynb \
  >> /mnt/c/Users/Hemera/Projects/meep_1x8_progress/stage1_v2/nbconvert.log 2>&1

cp meep_1x8_splitter_v2.ipynb /mnt/c/Users/Hemera/Projects/meep_1x8_progress/
echo "DONE $(date)" >> /mnt/c/Users/Hemera/Projects/meep_1x8_progress/stage1_v2/nbconvert.log
