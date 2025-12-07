import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score, mean_squared_error
from datetime import datetime
import shap 

class ModelAnalyzer:
    def __init__(self, model_name, output_dir):
        self.model_name = model_name
        self.output_dir = output_dir
        self.start_time = None
        self.end_time = None
        self.history = {'train_loss': [], 'val_loss': [], 'train_mae': [], 'val_mae': [], 'epochs': []}
        os.makedirs(output_dir, exist_ok=True)

    def start_training(self):
        self.start_time = time.time()
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Training started...")

    def end_training(self):
        self.end_time = time.time()
        elapsed = self.end_time - self.start_time
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Training completed in {elapsed:.2f}s")
        return elapsed

    def log_epoch(self, epoch, train_loss=None, val_loss=None, train_mae=None, val_mae=None):
        self.history['epochs'].append(epoch)
        if train_loss is not None: self.history['train_loss'].append(train_loss)
        if val_loss is not None: self.history['val_loss'].append(val_loss)
        if train_mae is not None: self.history['train_mae'].append(train_mae)
        if val_mae is not None: self.history['val_mae'].append(val_mae)

    def compute_metrics(self, y_true, y_pred, prefix='val'):
        epsilon = 1e-8
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100
        
        return {
            f'{prefix}_mae': mean_absolute_error(y_true, y_pred),
            f'{prefix}_rmse': root_mean_squared_error(y_true, y_pred),
            f'{prefix}_mse': mean_squared_error(y_true, y_pred),
            f'{prefix}_r2': r2_score(y_true, y_pred),
            f'{prefix}_mape': mape,
        }

    def plot_training_history(self):
        if not self.history['epochs']:
            return

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        if self.history['train_loss'] or self.history['val_loss']:
            if self.history['train_loss']:
                axes[0].plot(self.history['epochs'], self.history['train_loss'], label='Train Loss')
            if self.history['val_loss']:
                axes[0].plot(self.history['epochs'], self.history['val_loss'], label='Val Loss')
            axes[0].set_title('Training & Validation Loss')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

        if self.history['train_mae'] or self.history['val_mae']:
            if self.history['train_mae']:
                axes[1].plot(self.history['epochs'], self.history['train_mae'], label='Train MAE')
            if self.history['val_mae']:
                axes[1].plot(self.history['epochs'], self.history['val_mae'], label='Val MAE')
            axes[1].set_title('Training & Validation MAE')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'training_history.png'), dpi=150)
        plt.close()

    def plot_predictions(self, y_true, y_pred, dataset_name='Validation'):
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        axes[0].scatter(y_true, y_pred, alpha=0.5)
        m = min(y_true.min(), y_pred.min())
        M = max(y_true.max(), y_pred.max())
        axes[0].plot([m, M], [m, M], 'r--')
        axes[0].set_title(f'{dataset_name}: Predictions vs True')
        axes[0].grid(True, alpha=0.3)

        residuals = y_true - y_pred
        axes[1].scatter(y_pred, residuals, alpha=0.5)
        axes[1].axhline(0, color='r', linestyle='--')
        axes[1].set_title(f'{dataset_name}: Residual Plot')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'predictions_{dataset_name.lower()}.png'), dpi=150)
        plt.close()

    def plot_error_distribution(self, y_true, y_pred, dataset_name='Validation'):
        errors = y_true - y_pred
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].hist(errors, bins=50, edgecolor='black')
        axes[0].set_title(f'{dataset_name}: Error Distribution')
        axes[0].grid(True, alpha=0.3)

        axes[1].boxplot(errors, vert=True)
        axes[1].set_title(f'{dataset_name}: Error Box Plot')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'error_distribution_{dataset_name.lower()}.png'), dpi=150)
        plt.close()

    def plot_feature_importance(self, model, feature_names, top_n=30):
        try:
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importances = np.abs(model.coef_)
                if importances.ndim > 1:
                    importances = importances.flatten()
            else:
                return

            if len(importances) != len(feature_names):
                print(f"Warning: Feature names ({len(feature_names)}) and importances ({len(importances)}) length mismatch.")
                return

            idx = np.argsort(importances)[::-1][:top_n]
            labels = [feature_names[i] for i in idx]
            vals = importances[idx]

            plt.figure(figsize=(10, 8))
            plt.barh(range(len(vals)), vals)
            plt.yticks(range(len(vals)), labels)
            plt.gca().invert_yaxis()
            plt.title('Feature Importance (Native)')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'feature_importance.png'), dpi=150)
            plt.close()
        except Exception as e:
            print(f"Could not plot native feature importance: {e}")

    def plot_shap_analysis(self, model, X, feature_names=None, sample_size=1000):
        
        model_type_str = str(type(model))
        is_rf = "Forest" in model_type_str or "RandomForest" in model_type_str
        
        if is_rf:
            limit = 200
            print(f"  [Info] Random Forest detected. Capping SHAP samples at {limit} to prevent memory crash.")
        else:
            limit = 1000 

        effective_sample_size = min(sample_size, limit)
        
        print(f"  Generating SHAP analysis using {effective_sample_size} samples...")
        
        try:
            if X.shape[0] > effective_sample_size:
                if isinstance(X, pd.DataFrame):
                    X_sample = X.sample(n=effective_sample_size, random_state=42)
                else:
                    idx = np.random.choice(X.shape[0], effective_sample_size, replace=False)
                    X_sample = X[idx]
            else:
                X_sample = X

            if isinstance(X_sample, np.ndarray) and feature_names:
                X_sample_df = pd.DataFrame(X_sample, columns=feature_names)
            else:
                X_sample_df = X_sample.copy()

            try:
                X_sample_df = X_sample_df.astype(float)
            except ValueError:
                X_sample_df = X_sample_df.apply(pd.to_numeric, errors='coerce').fillna(0)

            print("  - Calculating SHAP values...")
            
            shap_values = None

    
            if "XGB" in model_type_str or "Forest" in model_type_str or "Tree" in model_type_str or "LGBM" in model_type_str:
                try:
                    if hasattr(model, 'get_booster'): 
                        explainer = shap.TreeExplainer(model.get_booster())
                    else: 
                        explainer = shap.TreeExplainer(model)
                    
                    shap_values = explainer.shap_values(X_sample_df, check_additivity=False)
                except Exception as e:
                    print(f"  [Info] TreeExplainer failed ({e}), falling back to KernelExplainer...")

            elif "Linear" in model_type_str or "Ridge" in model_type_str or "Lasso" in model_type_str:
                try:
                    explainer = shap.LinearExplainer(model, X_sample_df)
                    shap_values = explainer.shap_values(X_sample_df)
                except Exception as e:
                    print(f"  [Info] LinearExplainer failed ({e}), falling back to KernelExplainer...")

            if shap_values is None:
                print("  [Info] Using KernelExplainer (Generic mode). This is slower but works for MLP/SVM.")
                explainer = shap.KernelExplainer(model.predict, X_sample_df, link="identity")
                shap_values = explainer.shap_values(X_sample_df, nsamples=effective_sample_size)

            plt.figure(figsize=(10, 8))
            
            if isinstance(shap_values, list):
                shap_values = shap_values[0] if len(shap_values) == 1 else shap_values[1]

            shap_values = np.array(shap_values)
            if shap_values.ndim == 3: 
                shap_values = shap_values[:, :, 0]

            shap.summary_plot(shap_values, X_sample_df, show=False, plot_size=(10, 8))
            plt.title(f"SHAP Summary Plot (N={effective_sample_size})")
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'shap_summary_beeswarm.png'), dpi=150, bbox_inches='tight')
            plt.close()

            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X_sample_df, plot_type="bar", show=False, plot_size=(10, 8))
            plt.title("SHAP Feature Importance (Bar)")
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'shap_importance_bar.png'), dpi=150, bbox_inches='tight')
            plt.close()
            
            print("  SHAP analysis saved.")

        except Exception as e:
            print(f"  [Error] Could not generate SHAP plots: {str(e)}")

    def save_summary_report(self, train_metrics, val_metrics, test_info, config):
        def default_converter(o):
            if isinstance(o, np.integer): return int(o)
            if isinstance(o, np.floating): return float(o)
            if isinstance(o, np.ndarray): return o.tolist()
            return str(o)

        report = {
            'model': self.model_name,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'training_time_seconds': self.end_time - self.start_time if self.end_time else None,
            'config': config,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'test_info': test_info,
        }

        with open(os.path.join(self.output_dir, 'report.json'), 'w') as f:
            json.dump(report, f, indent=2, default=default_converter)

    def generate_full_analysis(self, model, X_train, y_train, X_val, y_val, feature_names=None, config=None):
        print("Generating Full Analysis Report")
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)

        train_metrics = self.compute_metrics(y_train, y_train_pred, 'train')
        val_metrics = self.compute_metrics(y_val, y_val_pred, 'val')

        print("\nTraining Metrics:")
        for k, v in train_metrics.items(): print(f"  {k}: {v:.4f}")

        print("\nValidation Metrics:")
        for k, v in val_metrics.items(): print(f"  {k}: {v:.4f}")

        self.plot_training_history()
        self.plot_predictions(y_val, y_val_pred)
        self.plot_error_distribution(y_val, y_val_pred)
        
        if feature_names:
            self.plot_feature_importance(model, feature_names)
            self.plot_shap_analysis(model, X_val, feature_names=feature_names)

        self.save_summary_report(train_metrics, val_metrics, {'samples': X_val.shape[0]}, config or {})
        print("Analysis Complete!")

        return train_metrics, val_metrics
