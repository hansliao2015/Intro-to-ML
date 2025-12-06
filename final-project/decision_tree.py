import argparse
import os
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, learning_curve, validation_curve
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.preprocessing import OrdinalEncoder


# =========================================================
# 工具函式：seed
# =========================================================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)


# =========================================================
# 資料前處理
# =========================================================
def load_data(data_dir: str):
    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    return train_df, test_df


def preprocess(train_df: pd.DataFrame,
               test_df: pd.DataFrame,
               target_col: str = "price"):
    """沿用 main.py 的簡單前處理，但保留 feature_names 方便畫圖。

    流程：
    - 丟掉 target 為 NA 的列
    - 丟掉 'fullAddress'（自由文字）
    - y = price
    - 特徵 = 其它欄位（不含 price, ID）
    - 類別欄位：OrdinalEncoder
    - 數值欄位：中位數補缺失

    回傳：
        X_train: np.ndarray (n_train, n_features)
        y      : np.ndarray (n_train,)
        X_test : np.ndarray (n_test, n_features)
        train_ids, test_ids: 用於 submission
        feature_names: list[str]，對應 X 的欄位名稱（經過 encode 後的順序）
    """

    # 丟掉 target 為 NA 的樣本
    train_df = train_df.dropna(subset=[target_col]).reset_index(drop=True)

    # 丟掉 fullAddress（高基數自由文字）
    drop_cols = ["fullAddress"]
    for c in drop_cols:
        if c in train_df.columns:
            train_df = train_df.drop(columns=[c])
        if c in test_df.columns:
            test_df = test_df.drop(columns=[c])

    # 取得 ID
    train_ids = train_df["ID"] if "ID" in train_df.columns else train_df["Id"]
    test_ids = test_df["ID"] if "ID" in test_df.columns else test_df["Id"]

    # 目標與特徵
    y = train_df[target_col].values
    X = train_df.drop(columns=[target_col])

    # 保留 ID 以外當特徵
    feature_cols = [c for c in X.columns if c != "ID" and c != "Id"]
    X = X[feature_cols]
    X_test = test_df[feature_cols]

    # 拆成類別 / 數值欄位
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = [c for c in feature_cols if c not in cat_cols]

    # 類別欄位：OrdinalEncoder，fit 在 train，上 apply 到 train/test
    if cat_cols:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X_cat = enc.fit_transform(X[cat_cols].astype(str))
        X_test_cat = enc.transform(X_test[cat_cols].astype(str))
    else:
        X_cat = np.empty((len(X), 0))
        X_test_cat = np.empty((len(X_test), 0))

    # 數值欄位：中位數補缺失
    if num_cols:
        X_num = X[num_cols].copy()
        X_test_num = X_test[num_cols].copy()

        medians = X_num.median()
        X_num = X_num.fillna(medians)
        X_test_num = X_test_num.fillna(medians)

        X_num = X_num.values
        X_test_num = X_test_num.values
    else:
        X_num = np.empty((len(X), 0))
        X_test_num = np.empty((len(X_test), 0))

    # 組合數值與類別欄位
    X_train = np.hstack([X_num, X_cat])
    X_test = np.hstack([X_test_num, X_test_cat])

    # 建立對應的 feature_names（先數值欄位，再類別欄位）
    feature_names = num_cols + cat_cols

    return X_train, y, X_test, train_ids, test_ids, feature_names


# =========================================================
# 訓練 & 驗證
# =========================================================
def train_and_evaluate_decision_tree(
    X,
    y,
    max_depth: int,
    min_samples_split: int,
    val_ratio: float,
    seed: int,
):
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=val_ratio,
        random_state=seed,
    )

    model = DecisionTreeRegressor(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=seed,
    )

    model.fit(X_train, y_train)

    # === 計算 train MAE / RMSE ===
    train_pred = model.predict(X_train)
    train_mae = mean_absolute_error(y_train, train_pred)
    train_rmse = root_mean_squared_error(y_train, train_pred)

    # === validation MAE / RMSE ===
    val_pred = model.predict(X_val)
    val_mae = mean_absolute_error(y_val, val_pred)
    val_rmse = root_mean_squared_error(y_val, val_pred)

    print(f"[Decision Tree] Train MAE : {train_mae:.4f}")
    print(f"[Decision Tree] Train RMSE: {train_rmse:.4f}")
    print(f"[Decision Tree] Val   MAE : {val_mae:.4f}")
    print(f"[Decision Tree] Val   RMSE: {val_rmse:.4f}")

    train_size = X_train.shape[0]
    val_size = X_val.shape[0]

    # 回傳必要資訊，後面畫圖用
    return (
        model,
        train_mae,
        val_mae,
        train_rmse,
        val_rmse,
        train_size,
        val_size,
        y_val,
        val_pred,
    )


# =========================================================
# 在全部訓練資料上重新訓練（用來打 submission）
# =========================================================
def train_full_decision_tree(
    X,
    y,
    max_depth: int,
    min_samples_split: int,
    seed: int,
):
    model = DecisionTreeRegressor(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=seed,
    )
    model.fit(X, y)
    return model


# =========================================================
# 預測 test 並輸出 submission.csv
# =========================================================
def predict_and_save_submission(
    model,
    X_test,
    test_ids,
    output_dir: str,
    file_name: str = "pred.csv",
):
    preds = model.predict(X_test)

    submission = pd.DataFrame(
        {
            "ID": test_ids,
            "price": preds,
        }
    )

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, file_name)
    submission.to_csv(out_path, index=False)
    print(f"Submission saved to {out_path}")


# =========================================================
# 可視化：Feature Importance
# =========================================================
def plot_feature_importance(model, feature_names, output_dir, top_n: int = 30):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    # 只畫前 top_n 個特徵，避免太擠
    top_n = min(top_n, len(indices))
    top_idx = indices[:top_n]

    plt.figure(figsize=(10, 6))
    plt.bar(range(top_n), importances[top_idx])
    plt.xticks(range(top_n), [feature_names[i] for i in top_idx], rotation=90)
    plt.title("Feature Importance (Top {} Features)".format(top_n))
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "feature_importance.png")
    plt.savefig(out_path)
    plt.close()
    print(f"[INFO] Feature importance saved to {out_path}")


# =========================================================
# 可視化：Learning Curve
# =========================================================
def plot_learning_curve_decision_tree(args, X, y, output_dir):
    # 使用目前設定的 max_depth / min_samples_split
    est = DecisionTreeRegressor(
        max_depth=None if args.max_depth < 0 else args.max_depth,
        min_samples_split=args.min_samples_split,
        random_state=args.seed,
    )

    train_sizes, train_scores, val_scores = learning_curve(
        est,
        X,
        y,
        cv=3,
        scoring="neg_mean_absolute_error",
        train_sizes=np.linspace(0.1, 1.0, 5),
        shuffle=True,
        random_state=args.seed,
    )

    train_mae = -np.mean(train_scores, axis=1)
    val_mae = -np.mean(val_scores, axis=1)

    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, train_mae, marker="o", label="Train MAE")
    plt.plot(train_sizes, val_mae, marker="o", label="Validation MAE")
    plt.xlabel("Training Size")
    plt.ylabel("MAE")
    plt.title("Learning Curve (Decision Tree)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "learning_curve.png")
    plt.savefig(out_path)
    plt.close()
    print(f"[INFO] Learning curve saved to {out_path}")


# =========================================================
# 可視化：Validation Curve for max_depth
# =========================================================
def plot_validation_curve_max_depth(args, X, y, output_dir):
    # 這裡用一個 base estimator，max_depth 交給 validation_curve 掃
    est = DecisionTreeRegressor(
        min_samples_split=args.min_samples_split,
        random_state=args.seed,
    )

    # 掃描的 max_depth 範圍，可以視情況調整
    param_range = np.arange(2, 22, 2)

    train_scores, val_scores = validation_curve(
        est,
        X,
        y,
        param_name="max_depth",
        param_range=param_range,
        cv=3,
        scoring="neg_mean_absolute_error",
    )

    train_mae = -np.mean(train_scores, axis=1)
    val_mae = -np.mean(val_scores, axis=1)

    plt.figure(figsize=(8, 6))
    plt.plot(param_range, train_mae, marker="o", label="Train MAE")
    plt.plot(param_range, val_mae, marker="o", label="Validation MAE")
    plt.xlabel("max_depth")
    plt.ylabel("MAE")
    plt.title("Validation Curve (max_depth)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "validation_curve_max_depth.png")
    plt.savefig(out_path)
    plt.close()
    print(f"[INFO] Validation curve (max_depth) saved to {out_path}")


# =========================================================
# 可視化：Error Distribution
# =========================================================
def plot_error_distribution(y_true, y_pred, output_dir):
    errors = y_true - y_pred

    plt.figure(figsize=(8, 6))
    plt.hist(errors, bins=50)
    plt.xlabel("Error (y_true - y_pred)")
    plt.ylabel("Count")
    plt.title("Validation Error Distribution")
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "error_distribution.png")
    plt.savefig(out_path)
    plt.close()
    print(f"[INFO] Error distribution saved to {out_path}")


# =========================================================
# 新增：寫入 output/report.txt
# =========================================================
def save_report(output_dir,
                args,
                train_mae,
                val_mae,
                train_rmse,
                val_rmse,
                train_size,
                val_size,
                model_depth,
                n_leaves,
                n_features):
    report_path = os.path.join(output_dir, "report.txt")

    lines = []
    lines.append("=== Decision Tree Training Report ===\n")
    lines.append(f"max_depth           : {args.max_depth}")
    lines.append(f"min_samples_split   : {args.min_samples_split}")
    lines.append(f"val_ratio           : {args.val_ratio}")
    lines.append(f"seed                : {args.seed}\n")
    lines.append(f"Train samples       : {train_size}")
    lines.append(f"Validation samples  : {val_size}")
    lines.append(f"Number of features  : {n_features}\n")
    lines.append(f"Tree depth          : {model_depth}")
    lines.append(f"Number of leaves    : {n_leaves}\n")
    lines.append(f"Train MAE           : {train_mae:.4f}")
    lines.append(f"Train RMSE          : {train_rmse:.4f}")
    lines.append(f"Validation MAE      : {val_mae:.4f}")
    lines.append(f"Validation RMSE     : {val_rmse:.4f}")
    lines.append("\n==============================\n")

    os.makedirs(output_dir, exist_ok=True)
    with open(report_path, "w") as f:
        f.write("\n".join(lines))

    print(f"[INFO] Report saved to {report_path}")


# =========================================================
# main
# =========================================================
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
    )

    parser.add_argument("--max_depth", type=int, default=15)
    parser.add_argument("--min_samples_split", type=int, default=5)

    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    set_seed(args.seed)

    print("=== 讀取資料 ===")
    train_df, test_df = load_data(args.data_dir)

    print("=== 前處理 ===")
    X, y, X_test, train_ids, test_ids, feature_names = preprocess(
        train_df, test_df, target_col="price"
    )

    print("=== 訓練 + 驗證 Decision Tree ===")
    dt_model, train_mae, val_mae, train_rmse, val_rmse, train_size, val_size, y_val, val_pred = \
        train_and_evaluate_decision_tree(
            X=X,
            y=y,
            max_depth=None if args.max_depth < 0 else args.max_depth,
            min_samples_split=args.min_samples_split,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )

    print("=== 可視化：Feature Importance ===")
    plot_feature_importance(dt_model, feature_names, args.output_dir, top_n=30)

    print("=== 可視化：Learning Curve ===")
    plot_learning_curve_decision_tree(args, X, y, args.output_dir)

    print("=== 可視化：Validation Curve (max_depth) ===")
    plot_validation_curve_max_depth(args, X, y, args.output_dir)

    print("=== 可視化：Error Distribution (Validation) ===")
    plot_error_distribution(y_val, val_pred, args.output_dir)

    # Tree complexity
    model_depth = dt_model.get_depth()
    n_leaves = dt_model.get_n_leaves()
    n_features = X.shape[1]

    print("=== 在全部訓練資料上重新訓練（用來做 submission） ===")
    full_model = train_full_decision_tree(
        X=X,
        y=y,
        max_depth=None if args.max_depth < 0 else args.max_depth,
        min_samples_split=args.min_samples_split,
        seed=args.seed,
    )

    print("=== 儲存報告 ===")
    save_report(
        output_dir=args.output_dir,
        args=args,
        train_mae=train_mae,
        val_mae=val_mae,
        train_rmse=train_rmse,
        val_rmse=val_rmse,
        train_size=train_size,
        val_size=val_size,
        model_depth=model_depth,
        n_leaves=n_leaves,
        n_features=n_features,
    )

    print("=== 產生 submission ===")
    predict_and_save_submission(
        model=full_model,
        X_test=X_test,
        test_ids=test_ids,
        output_dir=args.output_dir,
        file_name="pred.csv",
    )


if __name__ == "__main__":
    main()
