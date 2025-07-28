import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
from src.data.pipeline import DataPipeline
from src.models.train_simple import ModelTrainer
from src.deployment.server_simple import ModelServer
from src.monitoring.drift_detector_simple import ModelMonitor
from src.utils.config import config
from src.utils.logger_simple import logger

def run_data_pipeline(data_path: str, target_column: str) -> Dict[str, Any]:
    logger.info("Starting data pipeline task")
    
    pipeline = DataPipeline()
    pipeline_info = pipeline.run_pipeline(data_path, target_column)
    
    logger.info("Data pipeline task completed")
    return pipeline_info

def run_model_training() -> Dict[str, Any]:
    logger.info("Starting model training task")
    
    trainer = ModelTrainer()
    training_results = trainer.run_training_pipeline()
    
    logger.info("Model training task completed")
    return training_results

def setup_monitoring(reference_data_path: str) -> ModelMonitor:
    logger.info("Setting up model monitoring")
    
    reference_data = pd.read_csv(reference_data_path)
    monitor = ModelMonitor(reference_data)
    
    logger.info("Model monitoring setup completed")
    return monitor

def deploy_model() -> ModelServer:
    logger.info("Deploying model")
    
    server = ModelServer()
    
    logger.info("Model deployment completed")
    return server

def run_ml_pipeline(
    data_path: str,
    target_column: str,
    enable_monitoring: bool = True,
    enable_deployment: bool = True
) -> Dict[str, Any]:
    logger.info("Starting ML Pipeline orchestration")
    
    pipeline_results = {}
    
    try:
        logger.info("Step 1: Data Pipeline")
        data_pipeline_info = run_data_pipeline(data_path, target_column)
        pipeline_results['data_pipeline'] = data_pipeline_info
        
        logger.info("Step 2: Model Training")
        training_results = run_model_training()
        pipeline_results['training'] = training_results
        
        if enable_monitoring:
            logger.info("Step 3: Setup Monitoring")
            reference_data_path = Path(config.get("data.processed_data_path")) / "train.csv"
            monitor = setup_monitoring(str(reference_data_path))
            pipeline_results['monitoring'] = {'status': 'setup_completed'}
        
        if enable_deployment:
            logger.info("Step 4: Model Deployment")
            try:
                server = deploy_model()
                pipeline_results['deployment'] = {'status': 'deployed'}
            except Exception as e:
                logger.warning(f"Model deployment skipped: {str(e)}")
                pipeline_results['deployment'] = {'status': 'skipped', 'reason': str(e)}
        
        logger.info("ML Pipeline orchestration completed successfully")
        return pipeline_results
        
    except Exception as e:
        logger.error(f"ML Pipeline orchestration failed: {str(e)}")
        raise e

class MLPipelineOrchestrator:
    def __init__(self):
        self.config = config
        self.data_pipeline = None
        self.model_trainer = None
        self.model_server = None
        self.model_monitor = None
        
    def run_full_pipeline(self, data_path: str, target_column: str) -> Dict[str, Any]:
        logger.info("Running full ML pipeline")
        
        return run_ml_pipeline(
            data_path=data_path,
            target_column=target_column,
            enable_monitoring=True,
            enable_deployment=True
        )
    
    def run_data_only(self, data_path: str, target_column: str) -> Dict[str, Any]:
        logger.info("Running data pipeline only")
        
        return run_ml_pipeline(
            data_path=data_path,
            target_column=target_column,
            enable_monitoring=False,
            enable_deployment=False
        )
    
    def run_training_only(self) -> Dict[str, Any]:
        logger.info("Running model training only")
        
        return run_model_training()
    
    def deploy_model_only(self) -> ModelServer:
        logger.info("Deploying model only")
        
        return deploy_model()
    
    def setup_monitoring_only(self, reference_data_path: str) -> ModelMonitor:
        logger.info("Setting up monitoring only")
        
        return setup_monitoring(reference_data_path)
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        status = {
            'data_pipeline': self.data_pipeline is not None,
            'model_trainer': self.model_trainer is not None,
            'model_server': self.model_server is not None,
            'model_monitor': self.model_monitor is not None
        }
        
        return status

def create_sample_data():
    logger.info("Creating sample data for demonstration")
    
    np.random.seed(42)
    n_samples = 1000
    
    sample_data = pd.DataFrame({
        'feature1': np.random.normal(0, 1, n_samples),
        'feature2': np.random.normal(0, 1, n_samples),
        'feature3': np.random.choice(['A', 'B', 'C'], n_samples),
        'feature4': np.random.exponential(1, n_samples),
        'feature5': np.random.uniform(0, 10, n_samples),
        'target': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    })
    
    data_path = Path(config.get("data.raw_data_path"))
    data_path.mkdir(parents=True, exist_ok=True)
    
    sample_data.to_csv(data_path / "sample_data.csv", index=False)
    
    logger.info(f"Sample data created: {data_path / 'sample_data.csv'}")
    return str(data_path / "sample_data.csv")

if __name__ == "__main__":
    orchestrator = MLPipelineOrchestrator()
    
    sample_data_path = create_sample_data()
    
    results = orchestrator.run_full_pipeline(
        data_path=sample_data_path,
        target_column="target"
    )
    
    print("Pipeline completed successfully!")
    print(f"Best model: {results['training']['best_model_name']}")
    print(f"Test accuracy: {results['training']['final_evaluation']['test_metrics']['accuracy']:.4f}") 