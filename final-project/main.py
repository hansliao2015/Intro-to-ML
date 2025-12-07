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
    parser.add_argument("--preprocessing", type=str, default="v1",
                        choices=["v0", "v1", "v2"])
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_analysis", action="store_true",
                        help="Disable detailed analysis and visualization")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    model_output_dir = os.path.join(args.output_dir, f"{args.model}_{args.preprocessing}")
    os.makedirs(model_output_dir, exist_ok=True)

    print(f"Model: {args.model.upper()} | Preprocessing: {args.preprocessing}")
    print(f"Output: {model_output_dir}")

    # [1/6] 載入資料
    print("\n[1/6] Loading data...")
    train_df, test_df = load_data(args.data_dir)
    print(f"  Train: {train_df.shape}, Test: {test_df.shape}")

    # [2/6] 前處理
    print(f"\n[2/6] Preprocessing (method: {args.preprocessing})...")
    X, y, X_test, train_ids, test_ids, feature_names = preprocess(
        train_df, test_df, method=args.preprocessing
    )
    print(f"  Features: {X.shape[1]}, Train: {X.shape[0]}, Test: {X_test.shape[0]}")

    use_log_target = (args.preprocessing == "v2")
    if use_log_target:
        print("  Note: target is log1p-transformed; metrics will be computed in original price scale")

    # [3/6] Train/Val split
    print(f"\n[3/6] Splitting data (val_ratio={args.val_ratio})...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=args.val_ratio, random_state=args.seed
    )
    print(f"  Train: {X_train.shape[0]}, Val: {X_val.shape[0]}")

    # 初始化分析器
    analyzer = ModelAnalyzer(args.model, model_output_dir)

    # [4/6] 訓練模型
    print(f"\n[4/6] Training {args.model.upper()} model...")
    model_module = get_model_module(args.model)

    analyzer.start_training()
    # 所有模型的介面必須是: return model, val_mae, val_rmse, extra_info
    model, val_mae_raw, val_rmse_raw, extra_info = model_module.train_and_evaluate(
        X_train, y_train, X_val, y_val
    )
    training_time = analyzer.end_training()

    # [5/6] 計算指標與分析
    print(f"\n[5/6] Computing metrics and generating analysis...")

    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)

    if use_log_target:
        y_train_original = np.expm1(y_train)
        y_val_original = np.expm1(y_val)
        y_train_pred_original = np.expm1(y_train_pred)
        y_val_pred_original = np.expm1(y_val_pred)

        y_train_pred_original = np.clip(y_train_pred_original, 0, None)
        y_val_pred_original = np.clip(y_val_pred_original, 0, None)

        train_metrics = analyzer.compute_metrics(y_train_original, y_train_pred_original, 'train')
        val_metrics = analyzer.compute_metrics(y_val_original, y_val_pred_original, 'val')

        y_val_for_plot = y_val_original
        y_val_pred_for_plot = y_val_pred_original
    else:
        train_metrics = analyzer.compute_metrics(y_train, y_train_pred, 'train')
        val_metrics = analyzer.compute_metrics(y_val, y_val_pred, 'val')

        y_val_for_plot = y_val
        y_val_pred_for_plot = y_val_pred

    print(f"\n  Validation Metrics:")
    print(f"    MAE  : {val_metrics['val_mae']:.4f}")
    print(f"    RMSE : {val_metrics['val_rmse']:.4f}")
    if 'val_mse' in val_metrics:
        print(f"    MSE  : {val_metrics['val_mse']:.4f}")
    print(f"    R²   : {val_metrics['val_r2']:.4f}")
    print(f"    MAPE : {val_metrics['val_mape']:.2f}%")

    config = {
        'model': args.model,
        'preprocessing': args.preprocessing,
        'val_ratio': args.val_ratio,
        'seed': args.seed,
        'n_features': X.shape[1],
        'train_samples': X_train.shape[0],
        'val_samples': X_val.shape[0],
        'training_time': f"{training_time:.2f}s",
    }
    if isinstance(extra_info, dict):
        config.update(extra_info)

    analyzer.save_summary_report(
        train_metrics,
        val_metrics,
        {'test_samples': X_test.shape[0]},
        config
    )

    if not args.no_analysis:
        print("  Generating visualizations...")
        analyzer.plot_predictions(y_val_for_plot, y_val_pred_for_plot)
        analyzer.plot_error_distribution(y_val_for_plot, y_val_pred_for_plot)
        analyzer.plot_training_history()

        if feature_names and args.model in ["dt", "rf", "xgb", "lgbm"]:
            analyzer.plot_feature_importance(model, feature_names)
            if hasattr(analyzer, "plot_shap_analysis"):
                analyzer.plot_shap_analysis(model, X_val, feature_names=feature_names)
        print("  ✓ Analysis complete")
    else:
        print("  Skipping visualizations (--no_analysis)")

    # [6/6] 預測 test set
    print(f"\n[6/6] Predicting test set...")
    preds = model_module.predict_test(model, X_test)

    if use_log_target:
        preds = np.expm1(preds)
        preds = np.clip(preds, 0, None)
        print("  Note: predictions inverse-transformed to original price scale")

    save_submission(test_ids, preds, model_output_dir, f"pred_{args.model}.csv")
    print("All done! Results saved to:", model_output_dir)


if __name__ == "__main__":
    main()
