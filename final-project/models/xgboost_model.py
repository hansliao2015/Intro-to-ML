import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error


def train_and_evaluate(X_train, y_train, X_val, y_val):
    # 保守參數，避免偏向高維特徵
    model = XGBRegressor(
        n_estimators=300,           # 適中的迭代次數
        max_depth=5,                # 較淺的樹，避免過擬合
        learning_rate=0.05,         # 保守的學習率
        subsample=0.8,              # 行採樣，增加泛化
        colsample_bytree=0.7,       # 列採樣 70%，對高維特徵更公平
        min_child_weight=3,         # 增加正則化
        reg_alpha=0.1,              # L1 正則化
        reg_lambda=1.0,             # L2 正則化
        random_state=42,
        tree_method="hist"
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_val)

    val_mae = mean_absolute_error(y_val, preds)
    val_rmse = root_mean_squared_error(y_val, preds)

    extra_info = {
        "n_estimators": 300,
        "max_depth": 5,
        "colsample_bytree": 0.7
    }

    return model, val_mae, val_rmse, extra_info


def predict_test(model, X_test):
    preds = model.predict(X_test)
    return np.round(preds).astype(int)
