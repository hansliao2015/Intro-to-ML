import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, root_mean_squared_error


def train_and_evaluate(X_train, y_train, X_val, y_val):
    # 保守參數，對各種特徵維度公平
    model = lgb.LGBMRegressor(
        n_estimators=300,           # 適中的迭代次數
        num_leaves=31,              # 預設值，較保守
        learning_rate=0.05,         # 保守的學習率
        max_depth=6,                # 限制深度
        min_child_samples=20,       # 葉節點最小樣本數
        feature_fraction=0.8,       # 每次迭代使用 80% 特徵
        bagging_fraction=0.8,       # 行採樣
        bagging_freq=5,             # 每 5 次迭代做一次採樣
        reg_alpha=0.1,              # L1 正則化
        reg_lambda=1.0,             # L2 正則化
        random_state=42,
        verbose=-1
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_val)

    val_mae = mean_absolute_error(y_val, preds)
    val_rmse = root_mean_squared_error(y_val, preds)

    extra_info = {
        "num_leaves": 31,
        "n_estimators": 300,
        "feature_fraction": 0.8
    }

    return model, val_mae, val_rmse, extra_info


def predict_test(model, X_test):
    preds = model.predict(X_test)
    return np.round(preds).astype(int)
