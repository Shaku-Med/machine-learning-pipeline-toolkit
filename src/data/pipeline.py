import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from src.utils.config import config
from src.utils.logger_simple import logger

class DataPipeline:
    def __init__(self):
        self.raw_data_path = Path(config.get("data.raw_data_path"))
        self.processed_data_path = Path(config.get("data.processed_data_path"))
        self.features_path = Path(config.get("data.features_path"))
        self.random_state = config.get("data.random_state", 42)
        
        self.processed_data_path.mkdir(parents=True, exist_ok=True)
        self.features_path.mkdir(parents=True, exist_ok=True)
        
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = []
        self.target_column = None
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        logger.info(f"Loading data: {file_path}")
        
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        if file_path.suffix == '.csv':
            data = pd.read_csv(file_path)
        elif file_path.suffix == '.parquet':
            data = pd.read_parquet(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        logger.info(f"Data loaded successfully, shape: {data.shape}")
        return data
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        logger.info(f"Validating data, shape: {data.shape}")
        
        if data.empty:
            logger.error("Data is empty")
            return False
        
        if data.isnull().sum().sum() > 0:
            logger.warning(f"Data contains missing values: {data.isnull().sum().sum()}")
        
        duplicate_rows = data.duplicated().sum()
        if duplicate_rows > 0:
            logger.warning(f"Data contains duplicate rows: {duplicate_rows}")
        
        logger.info("Data validation completed")
        return True
    
    def preprocess_data(self, data: pd.DataFrame, target_column: str) -> pd.DataFrame:
        logger.info(f"Preprocessing data, target_column: {target_column}")
        
        self.target_column = target_column
        
        data_clean = data.copy()
        
        data_clean = self._handle_missing_values(data_clean)
        data_clean = self._encode_categorical_features(data_clean)
        data_clean = self._remove_outliers(data_clean)
        
        self.feature_columns = [col for col in data_clean.columns if col != target_column]
        
        logger.info(f"Data preprocessing completed, features_count: {len(self.feature_columns)}")
        return data_clean
    
    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        categorical_columns = data.select_dtypes(include=['object']).columns
        
        for col in numeric_columns:
            if data[col].isnull().sum() > 0:
                data[col].fillna(data[col].median(), inplace=True)
        
        for col in categorical_columns:
            if data[col].isnull().sum() > 0:
                data[col].fillna(data[col].mode()[0], inplace=True)
        
        return data
    
    def _encode_categorical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        categorical_columns = data.select_dtypes(include=['object']).columns
        
        for col in categorical_columns:
            if col != self.target_column:
                le = LabelEncoder()
                data[col] = le.fit_transform(data[col].astype(str))
                self.label_encoders[col] = le
        
        return data
    
    def _remove_outliers(self, data: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            if col != self.target_column:
                z_scores = np.abs((data[col] - data[col].mean()) / data[col].std())
                data = data[z_scores < threshold]
        
        return data
    
    def split_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        logger.info("Splitting data into train/validation/test sets")
        
        X = data[self.feature_columns]
        y = data[self.target_column]
        
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, 
            test_size=config.get("data.test_split", 0.2),
            random_state=self.random_state,
            stratify=y if len(y.unique()) < 10 else None
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=config.get("data.validation_split", 0.2),
            random_state=self.random_state,
            stratify=y_temp if len(y_temp.unique()) < 10 else None
        )
        
        logger.info(f"Data split completed - train: {X_train.shape}, val: {X_val.shape}, test: {X_test.shape}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def scale_features(self, X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        logger.info("Scaling features")
        
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        
        X_val_scaled = pd.DataFrame(
            self.scaler.transform(X_val),
            columns=X_val.columns,
            index=X_val.index
        )
        
        X_test_scaled = pd.DataFrame(
            self.scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
        
        logger.info("Feature scaling completed")
        return X_train_scaled, X_val_scaled, X_test_scaled
    
    def save_processed_data(self, X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame,
                          y_train: pd.Series, y_val: pd.Series, y_test: pd.Series):
        logger.info("Saving processed data")
        
        train_data = pd.concat([X_train, y_train], axis=1)
        val_data = pd.concat([X_val, y_val], axis=1)
        test_data = pd.concat([X_test, y_test], axis=1)
        
        train_data.to_csv(self.processed_data_path / "train.csv", index=False)
        val_data.to_csv(self.processed_data_path / "validation.csv", index=False)
        test_data.to_csv(self.processed_data_path / "test.csv", index=False)
        
        logger.info("Processed data saved successfully")
    
    def run_pipeline(self, data_path: str, target_column: str) -> Dict[str, Any]:
        logger.info(f"Starting data pipeline - data_path: {data_path}, target_column: {target_column}")
        
        data = self.load_data(data_path)
        
        if not self.validate_data(data):
            raise ValueError("Data validation failed")
        
        processed_data = self.preprocess_data(data, target_column)
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(processed_data)
        X_train_scaled, X_val_scaled, X_test_scaled = self.scale_features(X_train, X_val, X_test)
        
        self.save_processed_data(X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test)
        
        pipeline_info = {
            "feature_columns": self.feature_columns,
            "target_column": self.target_column,
            "scaler": self.scaler,
            "label_encoders": self.label_encoders,
            "data_shapes": {
                "train": X_train_scaled.shape,
                "validation": X_val_scaled.shape,
                "test": X_test_scaled.shape
            }
        }
        
        logger.info("Data pipeline completed successfully")
        return pipeline_info

if __name__ == "__main__":
    pipeline = DataPipeline()
    
    sample_data = pd.DataFrame({
        'feature1': np.random.normal(0, 1, 1000),
        'feature2': np.random.normal(0, 1, 1000),
        'feature3': np.random.choice(['A', 'B', 'C'], 1000),
        'target': np.random.choice([0, 1], 1000)
    })
    
    sample_data.to_csv("data/raw/sample_data.csv", index=False)
    
    pipeline_info = pipeline.run_pipeline("data/raw/sample_data.csv", "target")
    print("Pipeline completed:", pipeline_info) 