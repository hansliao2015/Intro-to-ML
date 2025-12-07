import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def run_eda(train_df, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))
    train_df['price'].hist(bins=50)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'eda_price_hist.png'), dpi=150)
    plt.close()

    num_cols = train_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    corr = train_df[num_cols].corr()

    plt.figure(figsize=(12, 10))
    plt.imshow(corr, cmap='coolwarm', interpolation='nearest')
    plt.colorbar()
    plt.xticks(range(len(num_cols)), num_cols, rotation=90)
    plt.yticks(range(len(num_cols)), num_cols)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'eda_corr.png'), dpi=150)
    plt.close()

    stats = train_df.describe(include='all')
    stats.to_csv(os.path.join(output_dir, 'eda_summary.csv'))


if __name__ == "__main__":
    from preprocessing import load_data
    
    # 預設參數
    data_dir = "./data"
    output_dir = "./eda"
    
    train_df, _ = load_data(data_dir)
    run_eda(train_df, output_dir)