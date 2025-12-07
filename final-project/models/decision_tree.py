import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error


def train_and_evaluate(X_train, y_train, X_val, y_val):
    # 保守參數，避免過擬合高維特徵
    model = DecisionTreeRegressor(
        max_depth=10,              # 限制深度，避免過擬合
        min_samples_split=20,      # 需要更多樣本才分裂
        min_samples_leaf=10,       # 葉節點至少 10 個樣本
        random_state=42
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_val)

    val_mae = mean_absolute_error(y_val, preds)
    val_rmse = root_mean_squared_error(y_val, preds)

    extra_info = {
        "max_depth": 10,
        "min_samples_split": 20,
        "min_samples_leaf": 10,
        "actual_depth": int(model.get_depth()),
        "n_leaves": int(model.get_n_leaves())
    }

    return model, val_mae, val_rmse, extra_info


def predict_test(model, X_test):
    preds = model.predict(X_test)
    return np.round(preds).astype(int)
