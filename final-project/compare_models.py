#!/usr/bin/env python3
"""
compare_models.py - 比較多個模型結果的工具
"""
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt


def load_experiment_results(base_dir):
    """載入所有模型結果"""
    results = []
    
    if not os.path.exists(base_dir):
        print(f"Directory not found: {base_dir}")
        return results
    
    for model_name in os.listdir(base_dir):
        model_path = os.path.join(base_dir, model_name)
        if not os.path.isdir(model_path):
            continue
        
        # 找 report_{model}.txt 檔案
        report_path = os.path.join(model_path, f'report_{model_name}.txt')
        if not os.path.exists(report_path):
            print(f"Warning: No report found for {model_name}")
            continue
        
        try:
            # 讀取 txt 格式報告
            result = {'model': model_name}
            with open(report_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or ':' not in line:
                        continue
                    
                    key, value = line.split(':', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # 解析數值
                    if key == 'Validation MAE':
                        result['val_mae'] = float(value)
                    elif key == 'Validation RMSE':
                        result['val_rmse'] = float(value)
                    elif key == 'training_time':
                        # 移除 's' 單位
                        result['train_time'] = float(value.rstrip('s'))
                    elif key == 'preprocessing':
                        result['preprocessing'] = value
            
            if 'val_mae' in result:
                results.append(result)
            else:
                print(f"Warning: No metrics found for {model_name}")
        
        except Exception as e:
            print(f"Warning: Failed to load {model_name}: {e}")
    
    return results


def create_comparison_table(results, output_path):
    """建立比較表格"""
    df = pd.DataFrame(results)
    
    # 排序（按 val_mae）
    if 'val_mae' in df.columns:
        df = df.sort_values('val_mae')
    
    # 儲存 CSV
    df.to_csv(output_path, index=False, float_format='%.2f')
    print(f"\nComparison table saved: {output_path}")
    
    # 顯示結果
    print("\nModel Comparison (sorted by Val MAE):")
    print(df.to_string(index=False))
    
    return df


def plot_model_comparison(df, output_dir):
    """繪製模型比較圖"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. MAE 比較
    if 'model' in df.columns and 'val_mae' in df.columns:
        plt.figure(figsize=(10, 6))
        
        sorted_df = df.sort_values('val_mae')
        colors = ['green' if i == 0 else 'skyblue' for i in range(len(sorted_df))]
        
        plt.bar(range(len(sorted_df)), sorted_df['val_mae'], color=colors)
        plt.xticks(range(len(sorted_df)), sorted_df['model'])
        plt.ylabel('Validation MAE')
        plt.title('Model Comparison - Validation MAE')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        save_path = os.path.join(output_dir, 'model_comparison.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_path}")
    
    # 2. RMSE 比較
    if 'model' in df.columns and 'val_rmse' in df.columns:
        plt.figure(figsize=(10, 6))
        
        sorted_df = df.sort_values('val_rmse')
        colors = ['green' if i == 0 else 'lightcoral' for i in range(len(sorted_df))]
        
        plt.bar(range(len(sorted_df)), sorted_df['val_rmse'], color=colors)
        plt.xticks(range(len(sorted_df)), sorted_df['model'])
        plt.ylabel('Validation RMSE')
        plt.title('Model Comparison - Validation RMSE')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        save_path = os.path.join(output_dir, 'model_comparison_rmse.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_path}")
    
    # 3. Training time 比較
    if 'model' in df.columns and 'train_time' in df.columns:
        valid_df = df.dropna(subset=['train_time'])
        if len(valid_df) > 0:
            plt.figure(figsize=(10, 6))
            
            sorted_df = valid_df.sort_values('train_time')
            
            plt.bar(range(len(sorted_df)), sorted_df['train_time'], color='gold')
            plt.xticks(range(len(sorted_df)), sorted_df['model'])
            plt.ylabel('Training Time (seconds)')
            plt.title('Model Comparison - Training Time')
            plt.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            
            save_path = os.path.join(output_dir, 'training_time_comparison.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Compare model results')
    parser.add_argument('--results_dir', type=str, default='./output',
                       help='Directory containing model results')
    parser.add_argument('--output_dir', type=str, default='./comparison',
                       help='Directory to save comparison results')
    args = parser.parse_args()
    
    print("Model Comparison Tool")
    print(f"Reading results from: {args.results_dir}")
    
    # 載入結果
    results = load_experiment_results(args.results_dir)
    
    if not results:
        print("No results found!")
        return
    
    print(f"Found {len(results)} models\n")
    
    # 建立比較表格
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, 'comparison_summary.csv')
    df = create_comparison_table(results, csv_path)
    
    # 繪製比較圖
    print("\nGenerating plots...")
    plot_model_comparison(df, args.output_dir)
    
    print("\nDone!")


if __name__ == '__main__':
    main()
