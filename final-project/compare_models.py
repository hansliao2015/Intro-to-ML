#!/usr/bin/env python3
"""
compare_models.py - 比較多個模型結果的工具
比較 18 個實驗：6 models × 3 preprocessing methods
"""
import os
import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_experiment_results(base_dir):
    """載入所有模型結果（從 report.json）"""
    results = []
    
    if not os.path.exists(base_dir):
        print(f"Directory not found: {base_dir}")
        return results
    
    for exp_dir in os.listdir(base_dir):
        exp_path = os.path.join(base_dir, exp_dir)
        if not os.path.isdir(exp_path):
            continue
        
        # 讀取 report.json
        report_path = os.path.join(exp_path, 'report.json')
        if not os.path.exists(report_path):
            print(f"Warning: No report.json found in {exp_dir}")
            continue
        
        try:
            with open(report_path, 'r') as f:
                data = json.load(f)
            
            # 解析目錄名稱：model_preprocessing (例如 dt_v0, rf_v1)
            if '_' in exp_dir:
                model_short, preproc = exp_dir.rsplit('_', 1)
            else:
                model_short = exp_dir
                preproc = 'unknown'
            
            # 從 val_metrics 或 config 中提取資訊
            val_metrics = data.get('val_metrics', {})
            config = data.get('config', {})
            
            result = {
                'experiment': exp_dir,
                'model': data.get('model', model_short),
                'preprocessing': config.get('preprocessing', preproc),
                'val_mae': val_metrics.get('val_mae'),
                'val_rmse': val_metrics.get('val_rmse'),
                'val_mse': val_metrics.get('val_mse'),
                'val_r2': val_metrics.get('val_r2'),
                'val_mape': val_metrics.get('val_mape'),
                'train_time': data.get('training_time_seconds'),
            }
            
            # 添加訓練指標（可選）
            train_metrics = data.get('train_metrics', {})
            result['train_mae'] = train_metrics.get('train_mae')
            result['train_r2'] = train_metrics.get('train_r2')
            
            # 添加配置資訊（可選）
            if 'n_features' in config:
                result['n_features'] = config['n_features']
            
            results.append(result)
        
        except Exception as e:
            print(f"Warning: Failed to load {exp_dir}: {e}")
    
    return results


def create_comparison_table(results, output_path):
    """建立比較表格"""
    df = pd.DataFrame(results)
    
    # 排序（按 model 再按 preprocessing）
    if 'model' in df.columns and 'preprocessing' in df.columns:
        df = df.sort_values(['model', 'preprocessing'])
    
    # 選擇並排序輸出欄位
    output_cols = ['experiment', 'model', 'preprocessing', 
                   'val_mae', 'val_rmse', 'val_r2', 'val_mape',
                   'train_mae', 'train_r2',
                   'train_time', 'n_features']
    output_cols = [col for col in output_cols if col in df.columns]
    
    # 選擇重要欄位顯示
    display_cols = ['experiment', 'model', 'preprocessing', 'val_mae', 'val_rmse', 
                    'val_r2', 'train_time']
    display_cols = [col for col in display_cols if col in df.columns]
    
    # 儲存完整 CSV（按指定欄位順序）
    df[output_cols].to_csv(output_path, index=False, float_format='%.4f')
    print(f"\nComparison table saved: {output_path}")
    
    # 顯示結果
    print("\nModel Comparison (sorted by model & preprocessing):")
    if display_cols:
        print(df[display_cols].to_string(index=False))
    else:
        print(df.to_string(index=False))
    
    return df


def plot_model_comparison(df, output_dir):
    """繪製模型比較圖"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 設定樣式
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. MAE 比較（按實驗）
    if 'experiment' in df.columns and 'val_mae' in df.columns:
        plt.figure(figsize=(14, 6))
        
        sorted_df = df.sort_values('val_mae')
        colors = ['green' if i == 0 else 'skyblue' for i in range(len(sorted_df))]
        
        plt.bar(range(len(sorted_df)), sorted_df['val_mae'], color=colors)
        plt.xticks(range(len(sorted_df)), sorted_df['experiment'], rotation=45, ha='right')
        plt.ylabel('Validation MAE')
        plt.title('All Experiments - Validation MAE (18 experiments)')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        save_path = os.path.join(output_dir, 'all_experiments_mae.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_path}")
    
    # 2. 按 preprocessing 分組的 MAE 比較
    if 'model' in df.columns and 'preprocessing' in df.columns and 'val_mae' in df.columns:
        plt.figure(figsize=(12, 6))
        
        # Pivot data for grouped bar chart
        pivot_df = df.pivot(index='model', columns='preprocessing', values='val_mae')
        pivot_df.plot(kind='bar', width=0.8)
        
        plt.ylabel('Validation MAE')
        plt.title('Model Performance by Preprocessing Method')
        plt.xlabel('Model')
        plt.legend(title='Preprocessing', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3, axis='y')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        save_path = os.path.join(output_dir, 'mae_by_preprocessing.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_path}")
    
    # 3. Heatmap - MAE
    if 'model' in df.columns and 'preprocessing' in df.columns and 'val_mae' in df.columns:
        plt.figure(figsize=(8, 6))
        
        pivot_df = df.pivot(index='model', columns='preprocessing', values='val_mae')
        sns.heatmap(pivot_df, annot=True, fmt='.2f', cmap='YlOrRd', cbar_kws={'label': 'MAE'})
        
        plt.title('Validation MAE Heatmap')
        plt.xlabel('Preprocessing Method')
        plt.ylabel('Model')
        plt.tight_layout()
        
        save_path = os.path.join(output_dir, 'mae_heatmap.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_path}")
    
    # 4. R² 比較
    if 'experiment' in df.columns and 'val_r2' in df.columns:
        plt.figure(figsize=(14, 6))
        
        sorted_df = df.sort_values('val_r2', ascending=False)
        colors = ['green' if i == 0 else 'lightcoral' for i in range(len(sorted_df))]
        
        plt.bar(range(len(sorted_df)), sorted_df['val_r2'], color=colors)
        plt.xticks(range(len(sorted_df)), sorted_df['experiment'], rotation=45, ha='right')
        plt.ylabel('Validation R²')
        plt.title('All Experiments - Validation R² (higher is better)')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        save_path = os.path.join(output_dir, 'all_experiments_r2.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_path}")
    
    # 5. Training time 比較
    if 'experiment' in df.columns and 'train_time' in df.columns:
        valid_df = df.dropna(subset=['train_time'])
        if len(valid_df) > 0:
            plt.figure(figsize=(14, 6))
            
            sorted_df = valid_df.sort_values('train_time')
            
            plt.bar(range(len(sorted_df)), sorted_df['train_time'], color='gold')
            plt.xticks(range(len(sorted_df)), sorted_df['experiment'], rotation=45, ha='right')
            plt.ylabel('Training Time (seconds)')
            plt.title('Training Time Comparison')
            plt.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            
            save_path = os.path.join(output_dir, 'training_time_comparison.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Saved: {save_path}")


def print_best_models(df):
    """顯示最佳模型"""
    print("\nBest Models:")
    
    if 'val_mae' in df.columns:
        valid_mae = df.dropna(subset=['val_mae'])
        if len(valid_mae) > 0:
            best_mae = valid_mae.loc[valid_mae['val_mae'].idxmin()]
            print(f"  Best MAE: {best_mae['experiment']} (MAE: {best_mae['val_mae']:.2f}", end="")
            if 'val_r2' in best_mae and pd.notna(best_mae['val_r2']):
                print(f", R²: {best_mae['val_r2']:.4f}", end="")
            print(")")
    
    if 'val_r2' in df.columns:
        valid_r2 = df.dropna(subset=['val_r2'])
        if len(valid_r2) > 0:
            best_r2 = valid_r2.loc[valid_r2['val_r2'].idxmax()]
            print(f"  Best R²: {best_r2['experiment']} (R²: {best_r2['val_r2']:.4f}", end="")
            if 'val_mae' in best_r2 and pd.notna(best_r2['val_mae']):
                print(f", MAE: {best_r2['val_mae']:.2f}", end="")
            print(")")
    
    if 'train_time' in df.columns:
        valid_time = df.dropna(subset=['train_time'])
        if len(valid_time) > 0:
            fastest = valid_time.loc[valid_time['train_time'].idxmin()]
            print(f"  Fastest: {fastest['experiment']} ({fastest['train_time']:.2f}s", end="")
            if 'val_mae' in fastest and pd.notna(fastest['val_mae']):
                print(f", MAE: {fastest['val_mae']:.2f}", end="")
            print(")")


def main():
    parser = argparse.ArgumentParser(
        description='Compare all model and preprocessing combinations (18 experiments)'
    )
    parser.add_argument('--results_dir', type=str, default='./output',
                       help='Directory containing model results')
    parser.add_argument('--output_dir', type=str, default='./comparison',
                       help='Directory to save comparison results')
    args = parser.parse_args()
    
    print("Model Comparison Tool")
    print(f"Reading from: {args.results_dir}")
    
    # 載入結果
    results = load_experiment_results(args.results_dir)
    
    if not results:
        print("No results found!")
        return
    
    print(f"Found {len(results)} experiments")
    
    # 建立比較表格
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, 'comparison_summary.csv')
    df = create_comparison_table(results, csv_path)
    
    # 顯示最佳模型
    print_best_models(df)
    
    # 繪製比較圖
    print("\nGenerating visualizations...")
    plot_model_comparison(df, args.output_dir)
    
    print(f"\nComparison complete! Results saved in: {args.output_dir}")


if __name__ == '__main__':
    main()
