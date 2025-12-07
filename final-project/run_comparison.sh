#!/bin/bash

# 檢查 ablation 目錄是否存在
if [ ! -d "ablation" ]; then
    echo "Error: ablation directory not found!"
    echo "Please run models first using: bash run_all_models.sh"
    exit 1
fi

# 執行比較
python compare_models.py \
    --results_dir ./ablation \
    --output_dir ./comparison
