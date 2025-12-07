import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, StandardScaler


# =========================================================
# Load dataset
# =========================================================
def load_data(data_dir: str):
    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


# =========================================================
# v0 – Minimal numeric baseline
# =========================================================
def preprocess_v0(train_df, test_df, target_col="price"):
    train_df = train_df.dropna(subset=[target_col]).reset_index(drop=True)

    y = train_df[target_col].values
    train_ids = train_df["ID"]
    test_ids = test_df["ID"]

    # Only numeric features
    num_cols = train_df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    num_cols.remove(target_col)

    X_train = train_df[num_cols].copy()
    X_test = test_df[num_cols].copy()

    # median imputation
    medians = X_train.median()
    X_train = X_train.fillna(medians)
    X_test = X_test.fillna(medians)

    return X_train.values, y, X_test.values, train_ids, test_ids, num_cols



# =========================================================
# v1 – One-hot + Missing indicator (EDA-driven)
# =========================================================
def preprocess_v1(train_df, test_df, target_col="price"):

    train_df = train_df.dropna(subset=[target_col]).reset_index(drop=True)

    y = train_df[target_col].values
    train_ids = train_df["ID"]
    test_ids = test_df["ID"]

    # Drop high-cardinality textual fields
    drop_cols = ["fullAddress", "postcode"]
    train_df = train_df.drop(columns=[c for c in drop_cols if c in train_df], errors="ignore")
    test_df = test_df.drop(columns=[c for c in drop_cols if c in test_df], errors="ignore")

    X_train = train_df.drop(columns=[target_col])
    X_test = test_df.copy()

    full = pd.concat([X_train, X_test], axis=0)

    num_cols = full.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = full.select_dtypes(include=["object"]).columns.tolist()

    # Numeric missing → median + missing indicator
    for col in num_cols:
        full[col + "_was_missing"] = full[col].isnull().astype(int)
        full[col] = full[col].fillna(full[col].median())

    # Categorical missing → explicit category
    for col in cat_cols:
        full[col] = full[col].fillna("Missing")

    # One-hot encoding
    full = pd.get_dummies(full, columns=cat_cols)

    feature_names = list(full.columns)
    n_train = len(X_train)

    return full.iloc[:n_train].values, y, full.iloc[n_train:].values, train_ids, test_ids, feature_names



# =========================================================
# v2 – Ordinal Encoding + Numeric Scaling + LOG-TRANSFORM target
# =========================================================
def preprocess_v2(train_df, test_df, target_col="price"):

    train_df = train_df.dropna(subset=[target_col]).reset_index(drop=True)

    # --------- LOG TRANSFORM TARGET ----------
    # log1p(price) compresses skewness (EDA showed skewness = 22.8)
    y_raw = train_df[target_col].values
    y = np.log1p(y_raw)     # model will learn in log-space

    train_ids = train_df["ID"]
    test_ids = test_df["ID"]

    # Drop high-cardinality fields
    drop_cols = ["fullAddress", "postcode"]
    train_df = train_df.drop(columns=[c for c in drop_cols if c in train_df])
    test_df = test_df.drop(columns=[c for c in drop_cols if c in test_df])

    X_train = train_df.drop(columns=[target_col])
    X_test = test_df.copy()

    cat_cols = X_train.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()

    # --------- Ordinal Encoding (handles 100+ categories efficiently) ----------
    if cat_cols:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X_train_cat = enc.fit_transform(X_train[cat_cols].astype(str))
        X_test_cat = enc.transform(X_test[cat_cols].astype(str))
    else:
        X_train_cat = np.empty((len(X_train), 0))
        X_test_cat = np.empty((len(X_test), 0))

    # --------- Numeric imputation (median) ----------
    X_train_num = X_train[num_cols].fillna(X_train[num_cols].median()).values
    X_test_num = X_test[num_cols].fillna(X_train[num_cols].median()).values

    # --------- Scaling numeric data ----------
    scaler = StandardScaler()
    X_train_num = scaler.fit_transform(X_train_num)
    X_test_num = scaler.transform(X_test_num)

    # --------- Combine ----------
    X_train_final = np.hstack([X_train_num, X_train_cat])
    X_test_final = np.hstack([X_test_num, X_test_cat])

    feature_names = num_cols + cat_cols

    return X_train_final, y, X_test_final, train_ids, test_ids, feature_names



# =========================================================
# Wrapper
# =========================================================
def preprocess(train_df, test_df, target_col="price", method="v1"):
    if method == "v0":
        return preprocess_v0(train_df, test_df, target_col)
    elif method == "v1":
        return preprocess_v1(train_df, test_df, target_col)
    elif method == "v2":
        return preprocess_v2(train_df, test_df, target_col)
    else:
        raise ValueError(f"Unknown preprocessing method: {method}. Choose from: v0, v1, v2.")