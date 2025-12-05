import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import root_mean_squared_error


# 最小可行版本：
# 1. 讀取 train.csv / test.csv
# 2. 將類別欄位做簡單編碼 (OrdinalEncoder)
# 3. 缺失值用簡單填補
# 4. 用 DecisionTreeRegressor 訓練
# 5. 對 test 做預測，輸出 pred.csv (ID, price)


def load_data(train_path: str = "train.csv", test_path: str = "test.csv"):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def simple_preprocess(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """最簡單的前處理：
    - 丟掉 'fullAddress' （太自由文字，先不用）
    - 目標: price
    - 特徵: 其他欄位（ID, price 以外）
    - 類別欄位: 用 OrdinalEncoder
    - 數值欄位: 缺失值用中位數
    """
    # 不用 fullAddress，比較單純
    drop_cols = ["fullAddress"]
    for c in drop_cols:
        if c in train_df.columns:
            train_df = train_df.drop(columns=[c])
        if c in test_df.columns:
            test_df = test_df.drop(columns=[c])

    # 分開 X, y
    y = train_df["price"].values
    X = train_df.drop(columns=["price"])

    # 保留 ID 以外當特徵
    feature_cols = [c for c in X.columns if c != "ID"]
    X = X[feature_cols]

    test_ids = test_df["ID"].values
    X_test = test_df[feature_cols]

    # 依資料型態拆類別 / 數值
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = [c for c in feature_cols if c not in cat_cols]

    # 類別欄位：OrdinalEncoder，fit 在 train 上，transform 在 train/test
    if cat_cols:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X_cat = enc.fit_transform(X[cat_cols].astype(str))
        X_test_cat = enc.transform(X_test[cat_cols].astype(str))
    else:
        X_cat = np.empty((len(X), 0))
        X_test_cat = np.empty((len(X_test), 0))

    # 數值欄位：缺失值用中位數
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

    # 串起來： [數值, 類別]
    X_all = np.hstack([X_num, X_cat])
    X_test_all = np.hstack([X_test_num, X_test_cat])

    return X_all, y, X_test_all, test_ids


def train_and_eval(X, y, random_state: int = 42):
    """簡單切 train/val，看一下大概的 RMSE，確認模型工作正常"""
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    model = DecisionTreeRegressor(
        max_depth=12,  # 給一個適中的深度
        random_state=random_state,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    # 新版 sklearn 直接提供 root_mean_squared_error
    rmse = root_mean_squared_error(y_val, y_pred)
    print(f"Validation RMSE: {rmse:,.2f}")

    return model


def train_full_and_predict(X, y, X_test, random_state: int = 42):
    """在全部 training data 上重新訓練，然後對 test 預測"""
    model = DecisionTreeRegressor(
        max_depth=12,
        random_state=random_state,
    )
    model.fit(X, y)
    test_pred = model.predict(X_test)
    return test_pred


def main():
    # 1. 讀資料
    train_df, test_df = load_data()

    # 2. 最簡前處理
    X_all, y, X_test_all, test_ids = simple_preprocess(train_df, test_df)

    # 3. cut 一下 validation，看模型是不是正常工作
    print("== Quick validation to check model works ==")
    _ = train_and_eval(X_all, y)

    # 4. 用全訓練資料重訓，並對 test 預測
    print("== Train on full data and predict test ==")
    test_pred = train_full_and_predict(X_all, y, X_test_all)

    # 5. 存成 pred.csv (ID, price)
    out_df = pd.DataFrame({
        "ID": test_ids,
        "price": test_pred,
    })
    out_df.to_csv("pred.csv", index=False)
    print("Saved predictions to pred.csv")


if __name__ == "__main__":
    main()
