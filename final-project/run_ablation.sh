#!/bin/bash
# run_ablation_fast.sh - Run ablation study with progress tracking

MODELS=("dt" "rf" "xgb" "lgbm" "linear" "mlp")
PREPROCESSINGS=("v0" "v1" "v2")
OUTPUT_DIR="./ablation"

total=$((${#MODELS[@]} * ${#PREPROCESSINGS[@]}))
current=0
failed=0

echo "======================================"
echo "ABLATION STUDY"
echo "======================================"
echo "Models: ${MODELS[@]}"
echo "Preprocessing: ${PREPROCESSINGS[@]}"
echo "Total experiments: $total"
echo "Output: $OUTPUT_DIR"
echo ""

start_time=$(date +%s)

for preproc in "${PREPROCESSINGS[@]}"; do
    echo ""
    echo "--- Preprocessing: $preproc ---"
    for model in "${MODELS[@]}"; do
        current=$((current + 1))
        
        printf "[%2d/%2d] %-6s + %-3s ... " "$current" "$total" "$model" "$preproc"
        
        if python main.py \
            --model "$model" \
            --preprocessing "$preproc" \
            --output_dir "$OUTPUT_DIR" \
            --no_analysis \
            > /dev/null 2>&1; then
            echo "✓"
        else
            echo "✗ FAILED"
            failed=$((failed + 1))
        fi
    done
done

end_time=$(date +%s)
elapsed=$((end_time - start_time))

echo ""
echo "======================================"
echo "ABLATION COMPLETE!"
echo "======================================"
echo "Completed: $((total - failed))/$total experiments"
echo "Failed: $failed"
echo "Time: ${elapsed}s"
echo ""
echo "Results: $OUTPUT_DIR/"
echo ""
echo "Next: python compare_models.py --input_dir $OUTPUT_DIR"
