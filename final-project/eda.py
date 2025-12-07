import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json


def run_eda(train_df, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    print(f"Running EDA... Output directory: {output_dir}/")

    # Identify numerical / categorical features
    num_cols = train_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = train_df.select_dtypes(include=['object']).columns.tolist()
    if 'ID' in num_cols:
        num_cols.remove('ID')
    if 'price' in num_cols:
        num_cols.remove('price')

    print(f"Rows: {len(train_df):,}, Columns: {len(train_df.columns)}")
    print(f"Numerical: {len(num_cols)}, Categorical: {len(cat_cols)}")

    # ===== 1. Missing values =====
    missing = train_df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    missing_df = pd.DataFrame({
        'feature': missing.index,
        'missing_count': missing.values,
        'missing_percentage': (missing.values / len(train_df) * 100).round(2)
    }) if len(missing) else pd.DataFrame()

    # ===== 2. Target distribution =====
    price = train_df['price'].dropna()
    price_quantiles = {f"q{int(q*100)}": float(price.quantile(q)) for q in [0.25, 0.5, 0.75, 0.9, 0.95]}

    # ===== 3. Categorical feature summary =====
    cat_summary = []
    for col in cat_cols:
        vc = train_df[col].value_counts()
        n_unique = len(vc)
        top_pct = (vc.iloc[0] / len(train_df) * 100) if len(vc) > 0 else 0
        cat_summary.append({
            'feature': col,
            'unique_values': n_unique,
            'top_freq_pct': round(top_pct, 2)
        })

    # ===== 4. Numerical feature summary =====
    num_summary = []
    for col in num_cols:
        vals = train_df[col].dropna()
        num_summary.append({
            'feature': col,
            'mean': float(vals.mean()),
            'median': float(vals.median()),
            'std': float(vals.std()),
            'min': float(vals.min()),
            'max': float(vals.max()),
            'skewness': float(vals.skew())
        })

    # ===== 5. Correlation with price =====
    correlations = []
    for col in num_cols:
        c = train_df[['price', col]].corr().iloc[0, 1]
        correlations.append((col, float(0 if pd.isna(c) else c)))
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    # ===== 6. Safe correlation heatmap =====
    num_cols_for_heatmap = ['price'] + num_cols
    if len(num_cols_for_heatmap) <= 40:  # Prevent crash
        plt.figure(figsize=(12, 10))
        sns.heatmap(train_df[num_cols_for_heatmap].corr(),
                    annot=True, fmt=".2f", cmap="coolwarm", square=True)
        plt.title("Correlation Heatmap")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "eda_correlation_heatmap.png"), dpi=150)
        plt.close()
        print("  ✓ correlation heatmap")
    else:
        print(f"  Skipping heatmap (too many features: {len(num_cols_for_heatmap)})")

    # ===== 7. Boxplots for categorical features =====
    def safe_boxplot(df, group_col, filename, max_groups=10):
        if group_col not in df.columns:
            return

        vc = df[group_col].value_counts()
        groups = vc.index[:max_groups]

        data_to_plot = []
        labels = []

        for g in groups:
            subset = df[df[group_col] == g]['price'].dropna()
            if len(subset) >= 10:  # Avoid misleading plots
                data_to_plot.append(subset)
                labels.append(str(g))

        if len(data_to_plot) == 0:
            print(f"  Skipping boxplot for {group_col} (no valid data)")
            return

        plt.figure(figsize=(12, 6))
        plt.boxplot(data_to_plot)
        plt.xticks(range(1, len(labels)+1), labels, rotation=45, ha='right')
        plt.ylabel("Price")
        plt.title(f"Price distribution by {group_col}")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, filename), dpi=150)
        plt.close()
        print(f"  ✓ boxplot for {group_col}")

    safe_boxplot(train_df, 'propertyType', "eda_price_propertyType.png")
    safe_boxplot(train_df, 'bedrooms', "eda_price_bedrooms.png")
    safe_boxplot(train_df, 'region', "eda_price_region.png")

    # ===== 8. Missing values bar plot =====
    if len(missing):
        plt.figure(figsize=(10, 6))
        missing_pct = missing / len(train_df) * 100
        plt.barh(missing_pct.index, missing_pct.values, color='salmon')
        plt.xlabel("Missing %")
        plt.title("Missing Values")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'eda_missing_values.png'), dpi=150)
        plt.close()
        print("  ✓ missing values plot")

    # ===== 9. Outlier visualization =====
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.hist(price, bins=50, edgecolor='black')
    plt.title("Price Distribution")

    plt.subplot(1, 2, 2)
    plt.boxplot(price)
    plt.title("Price Outliers")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "eda_outliers.png"), dpi=150)
    plt.close()
    print("  ✓ outlier visualization")

    # ===== 10. Correlation bar chart =====
    plt.figure(figsize=(10, 6))
    names = [c[0] for c in correlations]
    vals = [c[1] for c in correlations]
    colors = ['green' if v > 0 else 'red' for v in vals]

    plt.barh(names, vals, color=colors)
    plt.axvline(0, color='black')
    plt.title("Feature Correlation with Price")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "eda_feature_correlation.png"), dpi=150)
    plt.close()
    print("  ✓ feature correlation chart")

    # ===== 11. JSON summary =====
    def convert(obj):
        if isinstance(obj, (pd.DataFrame, pd.Series)):
            return obj.to_dict(orient="records")
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    summary = {
        "dataset_info": {
            "rows": int(len(train_df)),
            "cols": int(len(train_df.columns)),
        },
        "missing_values": convert(missing_df),
        "target_stats": {
            "mean": float(price.mean()),
            "median": float(price.median()),
            "std": float(price.std()),
            "skewness": float(price.skew()),
            "quantiles": price_quantiles
        },
        "numerical_features": convert(num_summary),
        "categorical_features": convert(cat_summary),
        "correlations": [{"feature": f, "corr": c} for f, c in correlations]
    }

    with open(os.path.join(output_dir, "eda_analysis.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\nEDA complete!")
    print(f"Plots saved: {len([f for f in os.listdir(output_dir) if f.endswith('.png')])}")
    print(f"Summary saved to: eda_analysis.json\n")


if __name__ == "__main__":
    from preprocessing import load_data
    train_df, _ = load_data("./data")
    run_eda(train_df, "./eda")