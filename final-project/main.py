import argparse
import os
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from preprocessing import load_data, preprocess
from models import decision_tree, random_forest, xgboost_model, lightgbm_model, linear_regression, mlp
from analysis import ModelAnalyzer



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
    
    parser.add_argument("--no_analysis", action="store_true",
                       help="Disable detailed analysis and visualization")
    parser.add_argument("--save_predictions", action="store_true",
                       help="Save validation predictions to file")
    
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    
    model_output_dir = os.path.join(args.output_dir, args.model)
    os.makedirs(model_output_dir, exist_ok=True)
    
    print(f"Model: {args.model.upper()} | Preprocessing: {args.preprocessing}")
    print(f"Output: {model_output_dir}")

    # 載入資料
    print("\n[1/6] Loading data...")
    train_df, test_df = load_data(args.data_dir)
    print(f"  Train: {train_df.shape}, Test: {test_df.shape}")
    
    # 前處理
    print(f"\n[2/6] Preprocessing (method: {args.preprocessing})...")
    X, y, X_test, train_ids, test_ids, feature_names = preprocess(
        train_df, test_df, method=args.preprocessing
    )
    print(f"  Features: {X.shape[1]}, Train: {X.shape[0]}, Test: {X_test.shape[0]}")

    # Train/Val split
    print(f"\n[3/6] Splitting data (val_ratio={args.val_ratio})...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=args.val_ratio, random_state=args.seed
    )
    print(f"  Train: {X_train.shape[0]}, Val: {X_val.shape[0]}")

    # 初始化分析器
    analyzer = ModelAnalyzer(args.model, model_output_dir)
    
    # 訓練模型
    print(f"\n[4/6] Training {args.model.upper()} model...")
    model_module = get_model_module(args.model)
    
    analyzer.start_training()
    model, val_mae, val_rmse, extra_info = model_module.train_and_evaluate(
        X_train, y_train, X_val, y_val
    )
    training_time = analyzer.end_training()
    
    print(f"\n  Validation MAE : {val_mae:.4f}")
    print(f"  Validation RMSE: {val_rmse:.4f}")

    # 儲存基本報告
    extra_info['training_time'] = f"{training_time:.2f}s"
    extra_info['preprocessing'] = args.preprocessing
    extra_info['train_samples'] = X_train.shape[0]
    extra_info['val_samples'] = X_val.shape[0]
    save_report(model_output_dir, args.model, val_mae, val_rmse, extra_info)

    # 完整分析（如果啟用）
    if not args.no_analysis:
        print("\n[5/6] Generating analysis and visualizations...")
        config = {
            'model': args.model,
            'preprocessing': args.preprocessing,
            'val_ratio': args.val_ratio,
            'seed': args.seed,
            'n_features': X.shape[1],
        }
        
        train_metrics, val_metrics = analyzer.generate_full_analysis(
            model, X_train, y_train, X_val, y_val, 
            feature_names=feature_names, 
            config=config
        )
    else:
        print("\n[5/6] Skipping detailed analysis (--no_analysis)")

    # 預測 test set
    print(f"\n[6/6] Predicting test set...")
    preds = model_module.predict_test(model, X_test)
    save_submission(test_ids, preds, model_output_dir, f"pred_{args.model}.csv")
    
    print("All done! Results saved to:", model_output_dir)


if __name__ == "__main__":
    main()
