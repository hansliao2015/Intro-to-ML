import argparse
import os
import random

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, root_mean_squared_error


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
    # 保險起見：丟掉 target 為 NA 的列
    train_df = train_df.dropna(subset=[target_col]).reset_index(drop=True)
    train_ids = train_df["ID"] if "ID" in train_df.columns else train_df["Id"]
    test_ids = test_df["ID"] if "ID" in test_df.columns else test_df["Id"]

    y = train_df[target_col].values

    train_features = train_df.drop(columns=[target_col])

    # 高基數欄位丟棄
    drop_cols = []
    for c in ["fullAddress", "postcode"]:
        if c in train_features.columns:
            drop_cols.append(c)

    if drop_cols:
        train_features = train_features.drop(columns=drop_cols)
        test_df = test_df.drop(columns=drop_cols, errors="ignore")

    # 合併 encode
    full = pd.concat([train_features, test_df], axis=0, ignore_index=True)

    # 數值補中位數
    num_cols = full.select_dtypes(include=["int64", "float64"]).columns.tolist()
    for col in num_cols:
        full[col] = full[col].fillna(full[col].median())

    # 類別補 Missing
    cat_cols = full.select_dtypes(include=["object"]).columns.tolist()
    for col in cat_cols:
        full[col] = full[col].fillna("Missing")

    full = pd.get_dummies(full, columns=cat_cols, dummy_na=False)

    n_train = train_df.shape[0]
    X_train = full.iloc[:n_train, :].values
    X_test = full.iloc[n_train:, :].values

    return X_train, y, X_test, train_ids, test_ids


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

    # 回傳更多資訊
    return model, train_mae, val_mae, train_rmse, val_rmse, (X_train.shape[0], X_val.shape[0])


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
# 新增：寫入 output/report.txt
# =========================================================
def save_report(output_dir, args, train_mae, val_mae, train_rmse, val_rmse, train_size, val_size):
    report_path = os.path.join(output_dir, "report.txt")

    lines = []
    lines.append("=== Decision Tree Training Report ===\n")
    lines.append(f"max_depth           : {args.max_depth}")
    lines.append(f"min_samples_split   : {args.min_samples_split}")
    lines.append(f"val_ratio           : {args.val_ratio}")
    lines.append(f"seed                : {args.seed}\n")
    lines.append(f"Train samples       : {train_size}")
    lines.append(f"Validation samples  : {val_size}\n")
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
        default=".",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
    )

    parser.add_argument("--max_depth", type=int, default=10)
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
    X, y, X_test, train_ids, test_ids = preprocess(train_df, test_df, target_col="price")

    print("=== 訓練 + 驗證 Decision Tree ===")
    dt_model, train_mae, val_mae, train_rmse, val_rmse, (train_size, val_size) = train_and_evaluate_decision_tree(
        X=X,
        y=y,
        max_depth=None if args.max_depth < 0 else args.max_depth,
        min_samples_split=args.min_samples_split,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

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
