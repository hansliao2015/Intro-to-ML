import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error


def train_and_evaluate(X_train, y_train, X_val, y_val):
    # 保守參數，避免對高維特徵過度敏感
    model = MLPRegressor(
        hidden_layer_sizes=(128, 64),  # 適中的網路大小
        activation="relu",
        learning_rate_init=0.001,
        alpha=0.01,                    # L2 正則化，避免過擬合
        batch_size=128,                # 固定 batch size
        max_iter=500,                  # 增加訓練次數
        early_stopping=True,           # 早停避免過擬合
        validation_fraction=0.1,       # 內部驗證集
        n_iter_no_change=20,           # 20 次無改善就停止
        random_state=42
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_val)

    val_mae = mean_absolute_error(y_val, preds)
    val_rmse = root_mean_squared_error(y_val, preds)

    extra_info = {
        "hidden_layers": (128, 64),
        "alpha": 0.01,
        "early_stopping": True
    }

    return model, val_mae, val_rmse, extra_info


def predict_test(model, X_test):
    preds = model.predict(X_test)
    return np.round(preds).astype(int)
