#!/bin/bash

# run_all_models.sh - 跑所有模型和預處理方式的完整 ablation study

echo "=========================================="
echo "Running All Models x All Preprocessing"
echo "=========================================="
echo ""

# 設定參數
DATA_DIR="./data"
OUTPUT_DIR="./ablation"
SEED=42

# 要跑的模型列表
MODELS=("linear" "mlp" "dt" "rf" "xgb" "lgbm")

# 要跑的預處理方式
PREPROCESSINGS=("v0" "v1" "v2")

# 計數器
total_experiments=$((${#MODELS[@]} * ${#PREPROCESSINGS[@]}))
current=0
success_count=0
fail_count=0

echo "Total experiments: $total_experiments"
echo "Models: ${MODELS[*]}"
echo "Preprocessing: ${PREPROCESSINGS[*]}"
echo ""
echo "=========================================="
echo ""

# 雙層迴圈：每個模型 x 每種預處理
for preproc in "${PREPROCESSINGS[@]}"; do
    for model in "${MODELS[@]}"; do
        current=$((current + 1))
        
        echo "[$current/$total_experiments] Training: $model with preprocessing $preproc"
        
        python main.py \
            --model "$model" \
            --data_dir "$DATA_DIR" \
            --output_dir "$OUTPUT_DIR" \
            --preprocessing "$preproc" \
            --seed "$SEED" \
            --no_analysis
        
        if [ $? -eq 0 ]; then
            echo "  ✓ $model ($preproc) completed"
            success_count=$((success_count + 1))
        else
            echo "  ✗ $model ($preproc) failed"
            fail_count=$((fail_count + 1))
        fi
        echo ""
    done
done

echo "=========================================="
echo "All experiments completed!"
echo "=========================================="
echo "Total: $total_experiments"
echo "Success: $success_count"
echo "Failed: $fail_count"
echo ""
echo "Results saved in: $OUTPUT_DIR/"
echo "  Structure: {model}_{preprocessing}/"
echo "  Example: linear_v0/, dt_v1/, rf_v2/, etc."
echo ""
echo "To compare results:"
echo "  python compare_models.py --results_dir $OUTPUT_DIR"
