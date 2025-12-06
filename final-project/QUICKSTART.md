# Quick Start

## 準備資料集
- 將`train.csv`, `test.csv`, `sample_submission.csv`放到`./data`

## 訓練單一模型

```bash
# 基本用法
python main.py --model dt --preprocessing v2

# 快速模式（不生成圖表）
python main.py --model lgbm --no_analysis
```

輸出會自動分類到各模型目錄：
```
output/
├── dt/
│   ├── pred_dt.csv
│   ├── report_dt.txt
│   └── *.png
├── lgbm/
│   └── ...
```

## 訓練所有模型

```bash
bash run_all_models.sh
```

會依序訓練：dt, rf, xgb, lgbm, linear

## 比較結果

```bash
python compare_models.py --results_dir ./output
```

生成比較表格和圖表。
