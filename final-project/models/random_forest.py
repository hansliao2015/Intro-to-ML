import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error


def train_and_evaluate(X_train, y_train, X_val, y_val):
    # 保守參數，對各種特徵數量公平
    model = RandomForestRegressor(
        n_estimators=100,          # 適中的樹數量
        max_depth=15,              # 限制深度，避免過擬合高維特徵
        min_samples_split=10,      # 保守的分裂條件
        min_samples_leaf=5,        # 葉節點至少 5 個樣本
        max_features='sqrt',       # 使用 sqrt(n_features)，對高維更公平
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_val)

    val_mae = mean_absolute_error(y_val, preds)
    val_rmse = root_mean_squared_error(y_val, preds)

    extra_info = {
        "n_estimators": 100,
        "max_depth": 15,
        "max_features": "sqrt",
        "feature_importance": True
    }

    return model, val_mae, val_rmse, extra_info


def predict_test(model, X_test):
    preds = model.predict(X_test)
    return np.round(preds).astype(int)
