import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error


def train_and_evaluate(X_train, y_train, X_val, y_val):
    model = RandomForestRegressor(
        n_estimators=100,          # 降低樹數，加速
        max_depth=20,              # 限制最大深度，加速 & 減少 overfitting
        min_samples_split=5,       # 減少深度
        min_samples_leaf=2,        # 控制葉子數
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_val)

    val_mae = mean_absolute_error(y_val, preds)
    val_rmse = root_mean_squared_error(y_val, preds)

    extra_info = {
        "n_estimators": 300,
        "feature_importance": True
    }

    return model, val_mae, val_rmse, extra_info


def predict_test(model, X_test):
    preds = model.predict(X_test)
    return np.round(preds).astype(int)
