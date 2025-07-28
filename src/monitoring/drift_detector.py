import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from scipy import stats
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from collections import deque
import time
from src.utils.config import config
from src.utils.logger import logger

class DriftDetector:
    def __init__(self, reference_data: pd.DataFrame, window_size: int = 1000):
        self.reference_data = reference_data
        self.window_size = window_size
        self.drift_threshold = config.get("monitoring.drift_threshold", 0.1)
        
        self.feature_columns = reference_data.columns
        self.reference_stats = self._calculate_reference_stats()
        
        self.prediction_history = deque(maxlen=window_size)
        self.feature_history = deque(maxlen=window_size)
        self.performance_history = deque(maxlen=window_size)
        
        self.drift_alerts = []
        self.last_check_time = time.time()
    
    def _calculate_reference_stats(self) -> Dict[str, Dict[str, float]]:
        logger.info("Calculating reference data statistics")
        
        stats_dict = {}
        
        for column in self.feature_columns:
            if self.reference_data[column].dtype in ['int64', 'float64']:
                stats_dict[column] = {
                    'mean': float(self.reference_data[column].mean()),
                    'std': float(self.reference_data[column].std()),
                    'min': float(self.reference_data[column].min()),
                    'max': float(self.reference_data[column].max()),
                    'q25': float(self.reference_data[column].quantile(0.25)),
                    'q75': float(self.reference_data[column].quantile(0.75))
                }
            else:
                value_counts = self.reference_data[column].value_counts(normalize=True)
                stats_dict[column] = {
                    'value_distribution': value_counts.to_dict()
                }
        
        logger.info("Reference statistics calculated", features_count=len(stats_dict))
        return stats_dict
    
    def add_prediction(self, features: pd.Series, prediction: int, actual: Optional[int] = None):
        self.feature_history.append(features)
        self.prediction_history.append(prediction)
        
        if actual is not None:
            self.performance_history.append(actual == prediction)
    
    def detect_data_drift(self, current_data: pd.DataFrame) -> Dict[str, Any]:
        logger.info("Detecting data drift")
        
        drift_results = {
            'overall_drift': False,
            'feature_drifts': {},
            'drift_score': 0.0
        }
        
        total_drift_score = 0.0
        drift_count = 0
        
        for column in self.feature_columns:
            if column in current_data.columns:
                drift_info = self._detect_feature_drift(column, current_data[column])
                drift_results['feature_drifts'][column] = drift_info
                
                if drift_info['is_drifted']:
                    drift_count += 1
                    total_drift_score += drift_info['drift_score']
        
        if len(self.feature_columns) > 0:
            drift_results['drift_score'] = total_drift_score / len(self.feature_columns)
            drift_results['overall_drift'] = drift_results['drift_score'] > self.drift_threshold
        
        logger.info("Data drift detection completed", 
                   overall_drift=drift_results['overall_drift'],
                   drift_score=drift_results['drift_score'])
        
        return drift_results
    
    def _detect_feature_drift(self, column: str, current_values: pd.Series) -> Dict[str, Any]:
        ref_stats = self.reference_stats[column]
        
        if 'mean' in ref_stats:
            return self._detect_numeric_drift(column, current_values, ref_stats)
        else:
            return self._detect_categorical_drift(column, current_values, ref_stats)
    
    def _detect_numeric_drift(self, column: str, current_values: pd.Series, ref_stats: Dict[str, float]) -> Dict[str, Any]:
        current_mean = current_values.mean()
        current_std = current_values.std()
        
        mean_diff = abs(current_mean - ref_stats['mean']) / ref_stats['std']
        std_diff = abs(current_std - ref_stats['std']) / ref_stats['std']
        
        ks_statistic, p_value = stats.ks_2samp(
            self.reference_data[column].dropna(),
            current_values.dropna()
        )
        
        drift_score = (mean_diff + std_diff + ks_statistic) / 3
        is_drifted = drift_score > self.drift_threshold or p_value < 0.05
        
        return {
            'is_drifted': is_drifted,
            'drift_score': drift_score,
            'ks_statistic': ks_statistic,
            'p_value': p_value,
            'mean_diff': mean_diff,
            'std_diff': std_diff,
            'current_stats': {
                'mean': current_mean,
                'std': current_std
            },
            'reference_stats': ref_stats
        }
    
    def _detect_categorical_drift(self, column: str, current_values: pd.Series, ref_stats: Dict[str, Any]) -> Dict[str, Any]:
        ref_dist = ref_stats['value_distribution']
        current_dist = current_values.value_counts(normalize=True).to_dict()
        
        all_categories = set(ref_dist.keys()) | set(current_dist.keys())
        
        total_diff = 0.0
        for category in all_categories:
            ref_prob = ref_dist.get(category, 0.0)
            current_prob = current_dist.get(category, 0.0)
            total_diff += abs(ref_prob - current_prob)
        
        drift_score = total_diff / 2
        is_drifted = drift_score > self.drift_threshold
        
        return {
            'is_drifted': is_drifted,
            'drift_score': drift_score,
            'current_distribution': current_dist,
            'reference_distribution': ref_dist
        }
    
    def detect_performance_drift(self, window_size: Optional[int] = None) -> Dict[str, Any]:
        if len(self.performance_history) < 10:
            return {
                'is_drifted': False,
                'insufficient_data': True,
                'message': 'Insufficient performance data for drift detection'
            }
        
        if window_size is None:
            window_size = min(len(self.performance_history), 100)
        
        recent_performance = list(self.performance_history)[-window_size:]
        recent_accuracy = np.mean(recent_performance)
        
        if len(self.performance_history) >= window_size * 2:
            baseline_performance = list(self.performance_history)[-window_size*2:-window_size]
            baseline_accuracy = np.mean(baseline_performance)
            
            performance_drift = baseline_accuracy - recent_accuracy
            is_drifted = performance_drift > self.drift_threshold
            
            return {
                'is_drifted': is_drifted,
                'performance_drift': performance_drift,
                'recent_accuracy': recent_accuracy,
                'baseline_accuracy': baseline_accuracy,
                'insufficient_data': False
            }
        else:
            return {
                'is_drifted': False,
                'insufficient_data': True,
                'message': 'Insufficient baseline data for performance drift detection'
            }
    
    def detect_concept_drift(self, X: pd.DataFrame, y: pd.Series, model) -> Dict[str, Any]:
        logger.info("Detecting concept drift")
        
        if len(X) < 50:
            return {
                'is_drifted': False,
                'insufficient_data': True,
                'message': 'Insufficient data for concept drift detection'
            }
        
        predictions = model.predict(X)
        
        accuracy = accuracy_score(y, predictions)
        precision = precision_score(y, predictions, average='weighted', zero_division=0)
        recall = recall_score(y, predictions, average='weighted', zero_division=0)
        f1 = f1_score(y, predictions, average='weighted', zero_division=0)
        
        performance_metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
        if len(self.performance_history) > 0:
            baseline_accuracy = np.mean(list(self.performance_history)[-100:])
            concept_drift = baseline_accuracy - accuracy
            is_drifted = concept_drift > self.drift_threshold
        else:
            concept_drift = 0.0
            is_drifted = False
        
        return {
            'is_drifted': is_drifted,
            'concept_drift': concept_drift,
            'performance_metrics': performance_metrics,
            'insufficient_data': False
        }
    
    def get_drift_summary(self) -> Dict[str, Any]:
        current_time = time.time()
        
        summary = {
            'last_check_time': self.last_check_time,
            'time_since_last_check': current_time - self.last_check_time,
            'total_predictions': len(self.prediction_history),
            'total_alerts': len(self.drift_alerts),
            'recent_alerts': [alert for alert in self.drift_alerts 
                            if current_time - alert['timestamp'] < 3600],
            'feature_history_size': len(self.feature_history),
            'performance_history_size': len(self.performance_history)
        }
        
        if len(self.feature_history) > 0:
            recent_features = pd.DataFrame(list(self.feature_history)[-100:])
            data_drift = self.detect_data_drift(recent_features)
            summary['data_drift'] = data_drift
        
        if len(self.performance_history) > 0:
            performance_drift = self.detect_performance_drift()
            summary['performance_drift'] = performance_drift
        
        self.last_check_time = current_time
        return summary
    
    def add_drift_alert(self, alert_type: str, message: str, severity: str = "warning"):
        alert = {
            'type': alert_type,
            'message': message,
            'severity': severity,
            'timestamp': time.time()
        }
        
        self.drift_alerts.append(alert)
        logger.warning(f"Drift alert: {alert_type}", message=message, severity=severity)
    
    def reset_baseline(self, new_reference_data: pd.DataFrame):
        logger.info("Resetting drift detection baseline")
        
        self.reference_data = new_reference_data
        self.reference_stats = self._calculate_reference_stats()
        self.drift_alerts = []
        
        logger.info("Baseline reset completed")

class ModelMonitor:
    def __init__(self, reference_data: pd.DataFrame):
        self.drift_detector = DriftDetector(reference_data)
        self.monitoring_active = True
    
    def start_monitoring(self):
        logger.info("Starting model monitoring")
        self.monitoring_active = True
    
    def stop_monitoring(self):
        logger.info("Stopping model monitoring")
        self.monitoring_active = False
    
    def update_monitoring(self, features: pd.Series, prediction: int, actual: Optional[int] = None):
        if self.monitoring_active:
            self.drift_detector.add_prediction(features, prediction, actual)
    
    def check_drift(self, current_data: pd.DataFrame) -> Dict[str, Any]:
        return self.drift_detector.detect_data_drift(current_data)
    
    def get_monitoring_summary(self) -> Dict[str, Any]:
        return self.drift_detector.get_drift_summary() 