#!/bin/bash
# run_eda.sh - 只跑 EDA 的腳本

# 執行 EDA（只跑一次）
echo "Step 0: Running EDA..."
python eda.py

if [ $? -eq 0 ]; then
    echo "✓ EDA completed"
else
    echo "✗ EDA failed"
    exit 1
fi
echo ""