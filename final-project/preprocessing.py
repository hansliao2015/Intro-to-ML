# preprocessing.py
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder


def load_data(data_dir: str):
    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def preprocess_v1(train_df: pd.DataFrame,
                  test_df: pd.DataFrame,
                  target_col: str = "price"):
    """
    預處理方式 v1 (One-hot encoding + Missing Indicator):
      1. 丟掉 target 為 NA 的列
      2. 丟掉高基數欄位(fullAddress, postcode)
      3. 加 missing indicator，再補值
      4. one-hot encoding 類別
      5. 回傳 X_train, y, X_test, train_ids, test_ids, feature_names
    """

    train_df = train_df.dropna(subset=[target_col]).reset_index(drop=True)

    y = train_df[target_col].values
    train_ids = train_df["ID"]
    test_ids = test_df["ID"]

    train_features = train_df.drop(columns=[target_col])

    # 高基數欄位先丟棄
    drop_cols = []
    for c in ["fullAddress", "postcode"]:
        if c in train_features.columns:
            drop_cols.append(c)

    train_features = train_features.drop(columns=drop_cols, errors="ignore")
    test_df = test_df.drop(columns=drop_cols, errors="ignore")

    # 合併做 preprocessing
    full = pd.concat([train_features, test_df], axis=0, ignore_index=True)

    # 數值欄位：加 missing indicator + 補 median
    num_cols = full.select_dtypes(include=["int64", "float64"]).columns.tolist()
    for col in num_cols:
        full[col + "_was_missing"] = full[col].isnull().astype(int)
        full[col] = full[col].fillna(full[col].median())

    # 類別欄位：補 "Missing"
    cat_cols = full.select_dtypes(include=["object"]).columns.tolist()
    for col in cat_cols:
        full[col] = full[col].fillna("Missing")

    # one-hot encoding
    full = pd.get_dummies(full, columns=cat_cols, dummy_na=False)

    feature_names = full.columns.tolist()

    n_train = train_df.shape[0]
    X_train = full.iloc[:n_train, :].values
    X_test = full.iloc[n_train:, :].values

    return X_train, y, X_test, train_ids, test_ids, feature_names


def preprocess_v2(train_df: pd.DataFrame,
                  test_df: pd.DataFrame,
                  target_col: str = "price"):
    """
    預處理方式 v2 (Ordinal Encoding):
      1. 丟掉 target 為 NA 的列
      2. 丟掉高基數欄位(fullAddress)
      3. 類別欄位：OrdinalEncoder
      4. 數值欄位：中位數補缺失
      5. 回傳 X_train, y, X_test, train_ids, test_ids, feature_names
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


def preprocess(train_df: pd.DataFrame,
               test_df: pd.DataFrame,
               target_col: str = "price",
               method: str = "v1"):
    """
    統一預處理介面
    
    Args:
        train_df: 訓練資料
        test_df: 測試資料
        target_col: 目標欄位名稱
        method: 預處理方法 ("v1" 或 "v2")
            - "v1": One-hot encoding + Missing Indicator (適合 tree-based models)
            - "v2": Ordinal encoding (較簡單，特徵數較少)
    
    Returns:
        X_train, y, X_test, train_ids, test_ids, feature_names
    """
    if method == "v1":
        return preprocess_v1(train_df, test_df, target_col)
    elif method == "v2":
        return preprocess_v2(train_df, test_df, target_col)
    else:
        raise ValueError(f"Unknown preprocessing method: {method}. Choose 'v1' or 'v2'.")
