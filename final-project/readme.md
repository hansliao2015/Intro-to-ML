# Quick Start

## 準備資料集
- 將`train.csv`, `test.csv`, `sample_submission.csv`放到`./data`

## Exploratory Data Analysis (EDA)

```bash
bash run_eda.sh
```
- 結果會輸出到`./eda/`

## 訓練單一模型
- 參數可以自己指定，細節請看`main.py`
```bash
python main.py --model dt --preprocessing v1
```

## 進行preprocess的ablation study

```bash
bash run_ablation.sh
bash run_comparison.sh
```
