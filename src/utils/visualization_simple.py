import matplotlib.pyplot as plt
import seaborn as sns
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
        
        plt.style.use('default')
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
        
        sorted_indices = np.argsort(importance)[::-1]
        sorted_features = [features[i] for i in sorted_indices]
        sorted_importance = [importance[i] for i in sorted_indices]
        
        plt.barh(range(len(sorted_features)), sorted_importance)
        plt.yticks(range(len(sorted_features)), sorted_features)
        plt.xlabel('Feature Importance')
        plt.title('Feature Importance Analysis')
        plt.gca().invert_yaxis()
        
        if save:
            file_path = self.output_dir / "feature_importance.png"
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            logger.info(f"Feature importance plot saved: {file_path}")
        
        plt.show()
        return str(self.output_dir / "feature_importance.png")
    
    def plot_data_distribution(self, data: pd.DataFrame, save: bool = True) -> str:
        logger.info("Creating data distribution visualization")
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        n_cols = len(numeric_cols)
        
        if n_cols == 0:
            logger.warning("No numeric columns found for distribution plot")
            return ""
        
        n_rows = (n_cols + 2) // 3
        fig, axes = plt.subplots(n_rows, min(3, n_cols), figsize=(15, 5 * n_rows))
        
        if n_cols == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, col in enumerate(numeric_cols):
            row = i // 3
            col_idx = i % 3
            
            ax = axes[row, col_idx]
            ax.hist(data[col].dropna(), bins=30, alpha=0.7, edgecolor='black')
            ax.set_title(f'Distribution of {col}')
            ax.set_xlabel(col)
            ax.set_ylabel('Frequency')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            file_path = self.output_dir / "data_distribution.png"
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            logger.info(f"Data distribution plot saved: {file_path}")
        
        plt.show()
        return str(self.output_dir / "data_distribution.png")
    
    def plot_drift_detection(self, drift_results: Dict[str, Any], save: bool = True) -> str:
        logger.info("Creating drift detection visualization")
        
        if not drift_results:
            logger.warning("No drift results provided")
            return ""
        
        features = list(drift_results.keys())
        drift_scores = [drift_results[feature]['drift_score'] for feature in features]
        drift_detected = [drift_results[feature]['drift_detected'] for feature in features]
        
        plt.figure(figsize=(12, 6))
        
        colors = ['red' if detected else 'green' for detected in drift_detected]
        
        bars = plt.bar(range(len(features)), drift_scores, color=colors, alpha=0.7)
        plt.xlabel('Features')
        plt.ylabel('Drift Score')
        plt.title('Data Drift Detection Results')
        plt.xticks(range(len(features)), features, rotation=45)
        plt.axhline(y=0.05, color='red', linestyle='--', label='Drift Threshold (0.05)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        for i, (bar, detected) in enumerate(zip(bars, drift_detected)):
            if detected:
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                        'DRIFT', ha='center', va='bottom', color='red', fontweight='bold')
        
        plt.tight_layout()
        
        if save:
            file_path = self.output_dir / "drift_detection.png"
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            logger.info(f"Drift detection plot saved: {file_path}")
        
        plt.show()
        return str(self.output_dir / "drift_detection.png")
    
    def plot_roc_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray, save: bool = True) -> str:
        logger.info("Creating ROC curve visualization")
        
        from sklearn.metrics import roc_curve, auc
        
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
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
    
    def create_summary_report(self, training_results: Dict[str, Any], drift_results: Optional[Dict[str, Any]] = None) -> str:
        logger.info("Creating summary report")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('ML Pipeline Summary Report', fontsize=16)
        
        if training_results:
            best_model = training_results.get('best_model_name', 'Unknown')
            test_accuracy = training_results.get('final_evaluation', {}).get('test_metrics', {}).get('accuracy', 0)
            
            axes[0, 0].text(0.5, 0.5, f'Best Model: {best_model}\nTest Accuracy: {test_accuracy:.4f}', 
                           ha='center', va='center', transform=axes[0, 0].transAxes, fontsize=12)
            axes[0, 0].set_title('Model Performance')
            axes[0, 0].axis('off')
        
        if drift_results:
            drift_count = sum(1 for result in drift_results.values() if result.get('drift_detected', False))
            total_features = len(drift_results)
            
            axes[0, 1].text(0.5, 0.5, f'Drift Detected: {drift_count}/{total_features}\nFeatures', 
                           ha='center', va='center', transform=axes[0, 1].transAxes, fontsize=12)
            axes[0, 1].set_title('Data Drift Status')
            axes[0, 1].axis('off')
        
        plt.tight_layout()
        
        file_path = self.output_dir / "summary_report.png"
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        logger.info(f"Summary report saved: {file_path}")
        
        plt.show()
        return str(file_path) 