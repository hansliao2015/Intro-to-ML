import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def run_eda(train_df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Generating EDA visualizations in {output_dir}/")
    
    # 1. Correlation heatmap (只顯示數值特徵)
    num_cols = train_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if 'price' in num_cols:
        num_cols.remove('price')
        num_cols = ['price'] + num_cols  # price 放第一個
    
    corr = train_df[num_cols].corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, 
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title('EDA : correlation heatmap', fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'eda_correlation_heatmap.png'), dpi=150)
    plt.close()
    print("  ✓ correlation heatmap")
    
    # 2. Price distribution by propertyType
    if 'propertyType' in train_df.columns:
        plt.figure(figsize=(12, 6))
        property_types = train_df['propertyType'].value_counts().index[:5]  # 取前5種
        data_to_plot = [train_df[train_df['propertyType'] == pt]['price'].dropna() 
                        for pt in property_types]
        
        plt.boxplot(data_to_plot, labels=property_types)
        plt.ylabel('Price')
        plt.xlabel('Property Type')
        plt.title('EDA : price distribution by propertyType', fontsize=14, pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'eda_price_by_propertyType.png'), dpi=150)
        plt.close()
        print("  ✓ price distribution by propertyType")
    
    # 3. Price distribution by bedrooms
    if 'bedrooms' in train_df.columns:
        plt.figure(figsize=(12, 6))
        bedrooms_counts = train_df['bedrooms'].value_counts().sort_index()
        bedroom_vals = bedrooms_counts.index[:8]  # 取前8種
        
        data_to_plot = [train_df[train_df['bedrooms'] == br]['price'].dropna() 
                        for br in bedroom_vals]
        
        plt.boxplot(data_to_plot, labels=bedroom_vals)
        plt.ylabel('Price')
        plt.xlabel('Number of Bedrooms')
        plt.title('EDA : price distribution by bedrooms', fontsize=14, pad=20)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'eda_price_by_bedrooms.png'), dpi=150)
        plt.close()
        print("  ✓ price distribution by bedrooms")
    
    # 4. Price distribution by region/outcode
    if 'region' in train_df.columns:
        plt.figure(figsize=(14, 6))
        top_regions = train_df['region'].value_counts().index[:10]
        
        data_to_plot = [train_df[train_df['region'] == reg]['price'].dropna() 
                        for reg in top_regions]
        
        plt.boxplot(data_to_plot, labels=top_regions)
        plt.ylabel('Price')
        plt.xlabel('Region')
        plt.title('EDA : price distribution by region/outcode', fontsize=14, pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'eda_price_by_region.png'), dpi=150)
        plt.close()
        print("  ✓ price distribution by region/outcode")
    
    # 5. Missing values visualization
    plt.figure(figsize=(12, 6))
    missing = train_df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    
    if len(missing) > 0:
        missing_pct = (missing / len(train_df)) * 100
        
        plt.barh(range(len(missing)), missing_pct, color='coral')
        plt.yticks(range(len(missing)), missing.index)
        plt.xlabel('Missing Percentage (%)')
        plt.title('EDA : missing values visualization', fontsize=14, pad=20)
        plt.grid(True, alpha=0.3, axis='x')
        
        for i, v in enumerate(missing_pct):
            plt.text(v + 0.5, i, f'{v:.1f}%', va='center')
    else:
        plt.text(0.5, 0.5, 'No missing values', ha='center', va='center', fontsize=14)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'eda_missing_values.png'), dpi=150)
    plt.close()
    print("  ✓ missing values visualization")
    
    # 6. Outlier visualization (price)
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.hist(train_df['price'], bins=50, color='skyblue', edgecolor='black')
    plt.xlabel('Price')
    plt.ylabel('Frequency')
    plt.title('Price Distribution')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.boxplot(train_df['price'].dropna(), vert=True)
    plt.ylabel('Price')
    plt.title('Price Boxplot (Outliers)')
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('EDA : outlier visualization', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'eda_outlier_visualization.png'), dpi=150)
    plt.close()
    print("  ✓ outlier visualization")
    
    # Summary statistics
    stats = train_df.describe(include='all')
    stats.to_csv(os.path.join(output_dir, 'eda_summary.csv'))
    print(f"  ✓ summary statistics saved to eda_summary.csv")


if __name__ == "__main__":
    from preprocessing import load_data
    
    # 預設參數
    data_dir = "./data"
    output_dir = "./eda"
    
    train_df, _ = load_data(data_dir)
    run_eda(train_df, output_dir)