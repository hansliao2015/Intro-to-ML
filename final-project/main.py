import argparse
import os
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from preprocessing import load_data, preprocess
from models import decision_tree, random_forest, xgboost_model, lightgbm_model, linear_regression, mlp



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def save_submission(test_ids, preds, output_dir, file_name):
    os.makedirs(output_dir, exist_ok=True)
    preds = np.round(preds).astype(int)
    df = pd.DataFrame({"ID": test_ids, "price": preds})
    df.to_csv(os.path.join(output_dir, file_name), index=False)


def save_report(output_dir, model_name, val_mae, val_rmse, extra_info):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"report_{model_name}.txt")

    lines = []
    lines.append(f"Model: {model_name}")
    lines.append(f"Validation MAE: {val_mae:.4f}")
    lines.append(f"Validation RMSE: {val_rmse:.4f}")

    for key, value in extra_info.items():
        lines.append(f"{key}: {value}")

    with open(path, "w") as f:
        f.write("\n".join(lines))


def get_model_module(model_name):
    if model_name == "dt":
        return decision_tree
    elif model_name == "rf":
        return random_forest
    elif model_name == "xgb":
        return xgboost_model
    elif model_name == "lgbm":
        return lightgbm_model
    elif model_name == "linear":
        return linear_regression
    elif model_name == "mlp":
        return mlp
    else:
        raise ValueError(model_name)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--model", type=str, required=True,
                        choices=["linear", "mlp", "dt", "rf", "xgb", "lgbm"])
    parser.add_argument("--preprocessing", type=str, default="v1", choices=["v1", "v2"])
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    train_df, test_df = load_data(args.data_dir)
    X, y, X_test, train_ids, test_ids, feature_names = preprocess(
        train_df, test_df, method=args.preprocessing
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=args.val_ratio, random_state=args.seed
    )

    model_module = get_model_module(args.model)

    model, val_mae, val_rmse, extra_info = model_module.train_and_evaluate(
        X_train, y_train, X_val, y_val
    )

    save_report(args.output_dir, args.model, val_mae, val_rmse, extra_info)

    # if hasattr(model_module, "plot_feature_importance"):
    #     model_module.plot_feature_importance(model, feature_names, args.output_dir)

    # if hasattr(model_module, "plot_learning_curve"):
    #     model_module.plot_learning_curve(X, y, args.output_dir)

    preds = model_module.predict_test(model, X_test)

    save_submission(test_ids, preds, args.output_dir, f"pred_{args.model}.csv")


if __name__ == "__main__":
    main()
