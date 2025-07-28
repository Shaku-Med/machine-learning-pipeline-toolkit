import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from src.data.pipeline import DataPipeline
from src.utils.config import config

class TestDataPipeline:
    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'feature2': np.random.normal(0, 1, 100),
            'feature3': np.random.choice(['A', 'B', 'C'], 100),
            'target': np.random.choice([0, 1], 100)
        })
        return data
    
    @pytest.fixture
    def pipeline(self):
        return DataPipeline()
    
    def test_load_data(self, pipeline, sample_data, tmp_path):
        data_file = tmp_path / "test_data.csv"
        sample_data.to_csv(data_file, index=False)
        
        loaded_data = pipeline.load_data(str(data_file))
        
        assert loaded_data.shape == sample_data.shape
        assert list(loaded_data.columns) == list(sample_data.columns)
    
    def test_validate_data(self, pipeline, sample_data):
        assert pipeline.validate_data(sample_data) == True
        
        empty_data = pd.DataFrame()
        assert pipeline.validate_data(empty_data) == False
    
    def test_preprocess_data(self, pipeline, sample_data):
        processed_data = pipeline.preprocess_data(sample_data, 'target')
        
        assert 'target' in processed_data.columns
        assert len(pipeline.feature_columns) == 3
        assert pipeline.target_column == 'target'
        assert len(pipeline.label_encoders) > 0
    
    def test_split_data(self, pipeline, sample_data):
        processed_data = pipeline.preprocess_data(sample_data, 'target')
        X_train, X_val, X_test, y_train, y_val, y_test = pipeline.split_data(processed_data)
        
        assert len(X_train) > 0
        assert len(X_val) > 0
        assert len(X_test) > 0
        assert len(y_train) == len(X_train)
        assert len(y_val) == len(X_val)
        assert len(y_test) == len(X_test)
    
    def test_scale_features(self, pipeline, sample_data):
        processed_data = pipeline.preprocess_data(sample_data, 'target')
        X_train, X_val, X_test, y_train, y_val, y_test = pipeline.split_data(processed_data)
        
        X_train_scaled, X_val_scaled, X_test_scaled = pipeline.scale_features(X_train, X_val, X_test)
        
        assert X_train_scaled.shape == X_train.shape
        assert X_val_scaled.shape == X_val.shape
        assert X_test_scaled.shape == X_test.shape 