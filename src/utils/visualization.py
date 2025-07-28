import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from pathlib import Path
from src.utils.config import config
from src.utils.logger_simple import logger

class ModelVisualizer:
    def __init__(self, output_dir: str = "plots"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def plot_training_metrics(self, training_history: Dict[str, Any], save: bool = True) -> str:
        logger.info("Creating training metrics visualization")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Model Training Metrics Comparison', fontsize=16)
        
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
        
        for metric, pos in zip(metrics, positions):
            ax = axes[pos]
            
            model_names = []
            train_scores = []
            val_scores = []
            
            for model_name, results in training_history.items():
                model_names.append(model_name)
                train_scores.append(results['train_metrics'][metric])
                val_scores.append(results['val_metrics'][metric])
            
            x = np.arange(len(model_names))
            width = 0.35
            
            ax.bar(x - width/2, train_scores, width, label='Train', alpha=0.8)
            ax.bar(x + width/2, val_scores, width, label='Validation', alpha=0.8)
            
            ax.set_xlabel('Models')
            ax.set_ylabel(metric.capitalize())
            ax.set_title(f'{metric.capitalize()} Comparison')
            ax.set_xticks(x)
            ax.set_xticklabels(model_names, rotation=45)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            file_path = self.output_dir / "training_metrics.png"
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training metrics plot saved: {file_path}")
        
        plt.show()
        return str(self.output_dir / "training_metrics.png")
    
    def plot_confusion_matrix(self, confusion_matrix: np.ndarray, class_names: List[str] = None, save: bool = True) -> str:
        logger.info("Creating confusion matrix visualization")
        
        if class_names is None:
            class_names = [f'Class {i}' for i in range(len(confusion_matrix))]
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            confusion_matrix,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names
        )
        
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        if save:
            file_path = self.output_dir / "confusion_matrix.png"
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix plot saved: {file_path}")
        
        plt.show()
        return str(self.output_dir / "confusion_matrix.png")
    
    def plot_feature_importance(self, feature_importance: Dict[str, float], save: bool = True) -> str:
        logger.info("Creating feature importance visualization")
        
        features = list(feature_importance.keys())
        importance = list(feature_importance.values())
        
        plt.figure(figsize=(10, 6))
        bars = plt.barh(features, importance)
        
        plt.xlabel('Feature Importance')
        plt.title('Feature Importance Analysis')
        plt.gca().invert_yaxis()
        
        for i, bar in enumerate(bars):
            plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{importance[i]:.3f}', va='center')
        
        plt.grid(True, alpha=0.3)
        
        if save:
            file_path = self.output_dir / "feature_importance.png"
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            logger.info(f"Feature importance plot saved: {file_path}")
        
        plt.show()
        return str(self.output_dir / "feature_importance.png")
    
    def plot_data_distribution(self, data: pd.DataFrame, save: bool = True) -> str:
        logger.info("Creating data distribution visualization")
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        categorical_cols = data.select_dtypes(include=['object']).columns
        
        n_numeric = len(numeric_cols)
        n_categorical = len(categorical_cols)
        
        if n_numeric == 0 and n_categorical == 0:
            logger.warning("No numeric or categorical columns found for visualization")
            return ""
        
        fig, axes = plt.subplots(max(1, n_numeric), 2, figsize=(15, 5 * max(1, n_numeric)))
        if n_numeric == 1:
            axes = axes.reshape(1, -1)
        elif n_numeric == 0:
            axes = np.array([[plt.gca()]])
        
        for i, col in enumerate(numeric_cols):
            if i < len(axes):
                axes[i, 0].hist(data[col].dropna(), bins=30, alpha=0.7, edgecolor='black')
                axes[i, 0].set_title(f'{col} - Histogram')
                axes[i, 0].set_xlabel(col)
                axes[i, 0].set_ylabel('Frequency')
                
                axes[i, 1].boxplot(data[col].dropna())
                axes[i, 1].set_title(f'{col} - Box Plot')
                axes[i, 1].set_ylabel(col)
        
        plt.tight_layout()
        
        if save:
            file_path = self.output_dir / "data_distribution.png"
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            logger.info(f"Data distribution plot saved: {file_path}")
        
        plt.show()
        return str(self.output_dir / "data_distribution.png")
    
    def plot_drift_detection(self, drift_results: Dict[str, Any], save: bool = True) -> str:
        logger.info("Creating drift detection visualization")
        
        if 'feature_drifts' not in drift_results:
            logger.warning("No feature drift data available")
            return ""
        
        feature_drifts = drift_results['feature_drifts']
        
        features = list(feature_drifts.keys())
        drift_scores = [feature_drifts[feature]['drift_score'] for feature in features]
        is_drifted = [feature_drifts[feature]['is_drifted'] for feature in features]
        
        colors = ['red' if drifted else 'green' for drifted in is_drifted]
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(features, drift_scores, color=colors, alpha=0.7)
        
        plt.axhline(y=config.get("monitoring.drift_threshold", 0.1), 
                   color='red', linestyle='--', label='Drift Threshold')
        
        plt.xlabel('Features')
        plt.ylabel('Drift Score')
        plt.title('Feature Drift Detection')
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        for i, bar in enumerate(bars):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{drift_scores[i]:.3f}', ha='center', va='bottom')
        
        if save:
            file_path = self.output_dir / "drift_detection.png"
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            logger.info(f"Drift detection plot saved: {file_path}")
        
        plt.show()
        return str(self.output_dir / "drift_detection.png")
    
    def create_interactive_dashboard(self, training_results: Dict[str, Any], drift_results: Optional[Dict[str, Any]] = None) -> str:
        logger.info("Creating interactive dashboard")
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Model Performance', 'Feature Importance', 'Training Metrics', 'Drift Analysis'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "bar"}]]
        )
        
        if 'all_results' in training_results:
            model_names = list(training_results['all_results'].keys())
            f1_scores = [training_results['all_results'][name]['val_metrics']['f1'] 
                        for name in model_names]
            
            fig.add_trace(
                go.Bar(x=model_names, y=f1_scores, name='F1 Score'),
                row=1, col=1
            )
        
        if 'final_evaluation' in training_results and 'feature_importance' in training_results['final_evaluation']:
            feature_importance = training_results['final_evaluation']['feature_importance']
            if feature_importance:
                features = list(feature_importance.keys())
                importance = list(feature_importance.values())
                
                fig.add_trace(
                    go.Bar(x=features, y=importance, name='Feature Importance'),
                    row=1, col=2
                )
        
        if 'training_history' in training_results:
            history = training_results['training_history']
            for model_name, results in history.items():
                epochs = range(1, len(results.get('train_metrics', [])) + 1)
                train_acc = [results['train_metrics']['accuracy']] * len(epochs)
                val_acc = [results['val_metrics']['accuracy']] * len(epochs)
                
                fig.add_trace(
                    go.Scatter(x=epochs, y=train_acc, name=f'{model_name} Train'),
                    row=2, col=1
                )
                fig.add_trace(
                    go.Scatter(x=epochs, y=val_acc, name=f'{model_name} Val'),
                    row=2, col=1
                )
        
        if drift_results and 'feature_drifts' in drift_results:
            feature_drifts = drift_results['feature_drifts']
            features = list(feature_drifts.keys())
            drift_scores = [feature_drifts[feature]['drift_score'] for feature in features]
            
            fig.add_trace(
                go.Bar(x=features, y=drift_scores, name='Drift Score'),
                row=2, col=2
            )
        
        fig.update_layout(height=800, title_text="ML Pipeline Dashboard")
        
        file_path = self.output_dir / "interactive_dashboard.html"
        fig.write_html(str(file_path))
        
        logger.info(f"Interactive dashboard saved: {file_path}")
        return str(file_path)
    
    def plot_roc_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray, save: bool = True) -> str:
        logger.info("Creating ROC curve visualization")
        
        from sklearn.metrics import roc_curve, auc
        
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        if save:
            file_path = self.output_dir / "roc_curve.png"
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            logger.info(f"ROC curve plot saved: {file_path}")
        
        plt.show()
        return str(self.output_dir / "roc_curve.png") 