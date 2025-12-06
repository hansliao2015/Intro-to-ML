#!/bin/bash

# run_all_models.sh - 跑所有模型的簡單腳本

echo "Running All Models"
echo ""

# 設定參數
DATA_DIR="./data"
OUTPUT_DIR="./output"
PREPROCESSING="v2"  # 可改成 v1 或 v2
SEED=42

# 要跑的模型列表
MODELS=("dt" "rf" "xgb" "lgbm" "linear")

# 依序跑每個模型
for model in "${MODELS[@]}"; do
    echo "Training: $model"
    
    python main.py \
        --model "$model" \
        --data_dir "$DATA_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --preprocessing "$PREPROCESSING" \
        --seed "$SEED" \
        --no_analysis
    
    if [ $? -eq 0 ]; then
        echo "✓ $model completed"
    else
        echo "✗ $model failed"
    fi
    echo ""
done

echo "All models completed!"
echo "Results saved in: $OUTPUT_DIR"
echo ""
echo "To compare results:"
echo "  python compare_models.py --results_dir $OUTPUT_DIR"
