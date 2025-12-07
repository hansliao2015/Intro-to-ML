#!/bin/bash
# run_eda.sh - Run exploratory data analysis

echo "Running EDA..."
python eda.py

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ EDA completed successfully"
    echo "  Output files:"
    echo "    - eda/eda_analysis.json (structured data)"
    echo "    - eda/eda_summary.csv (statistics)"
    echo "    - eda/*.png (visualizations)"
    echo ""
    echo "Next: Review eda_analysis.json to design preprocessing strategies"
else
    echo "✗ EDA failed"
    exit 1
fi