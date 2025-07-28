import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import time
import psutil
import os
from src.utils.config import config
from src.utils.logger_simple import logger

class PredictionRequest(BaseModel):
    features: List[float] = Field(..., description="Input features for prediction")
    feature_names: Optional[List[str]] = Field(None, description="Feature names")

class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="Predicted class")
    probability: float = Field(..., description="Prediction probability")
    model_name: str = Field(..., description="Model used for prediction")
    timestamp: str = Field(..., description="Prediction timestamp")

class HealthResponse(BaseModel):
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    model_name: Optional[str] = Field(None, description="Loaded model name")
    uptime: float = Field(..., description="Service uptime in seconds")
    memory_usage: float = Field(..., description="Memory usage in MB")

class ModelServer:
    def __init__(self):
        self.app = FastAPI(
            title="ML Pipeline Model Server",
            description="RESTful API for ML model serving",
            version="1.0.0"
        )
        
        self.model_path = Path(config.get("model.model_path"))
        self.model_name = config.get("model.model_name")
        self.save_format = config.get("model.save_format")
        
        self.model = None
        self.model_name_loaded = None
        self.start_time = time.time()
        
        self.prediction_count = 0
        self.prediction_latencies = []
        
        self.setup_middleware()
        self.setup_routes()
        self.load_model()
    
    def setup_middleware(self):
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    def setup_routes(self):
        @self.app.get("/", response_model=Dict[str, str])
        async def root():
            return {"message": "ML Pipeline Model Server", "version": "1.0.0"}
        
        @self.app.get("/health", response_model=HealthResponse)
        async def health_check():
            return HealthResponse(
                status="healthy" if self.model is not None else "unhealthy",
                model_loaded=self.model is not None,
                model_name=self.model_name_loaded,
                uptime=time.time() - self.start_time,
                memory_usage=psutil.Process().memory_info().rss / 1024 / 1024
            )
        
        @self.app.get("/metrics")
        async def metrics():
            avg_latency = np.mean(self.prediction_latencies) if self.prediction_latencies else 0
            return {
                "predictions_total": self.prediction_count,
                "average_latency_seconds": avg_latency,
                "model_name": self.model_name_loaded
            }
        
        @self.app.post("/predict", response_model=PredictionResponse)
        async def predict(request: PredictionRequest, background_tasks: BackgroundTasks):
            if self.model is None:
                raise HTTPException(status_code=503, detail="Model not loaded")
            
            start_time = time.time()
            
            try:
                features = np.array(request.features).reshape(1, -1)
                
                if request.feature_names:
                    df = pd.DataFrame(features, columns=request.feature_names)
                else:
                    df = pd.DataFrame(features)
                
                prediction = self.model.predict(df)[0]
                probability = self.model.predict_proba(df)[0].max()
                
                response = PredictionResponse(
                    prediction=int(prediction),
                    probability=float(probability),
                    model_name=self.model_name_loaded,
                    timestamp=pd.Timestamp.now().isoformat()
                )
                
                latency = time.time() - start_time
                
                background_tasks.add_task(
                    self._log_prediction_metrics,
                    prediction,
                    latency
                )
                
                return response
                
            except Exception as e:
                logger.error(f"Prediction error: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
        
        @self.app.post("/predict_batch", response_model=List[PredictionResponse])
        async def predict_batch(requests: List[PredictionRequest]):
            if self.model is None:
                raise HTTPException(status_code=503, detail="Model not loaded")
            
            try:
                predictions = []
                
                for request in requests:
                    features = np.array(request.features).reshape(1, -1)
                    
                    if request.feature_names:
                        df = pd.DataFrame(features, columns=request.feature_names)
                    else:
                        df = pd.DataFrame(features)
                    
                    prediction = self.model.predict(df)[0]
                    probability = self.model.predict_proba(df)[0].max()
                    
                    predictions.append(PredictionResponse(
                        prediction=int(prediction),
                        probability=float(probability),
                        model_name=self.model_name_loaded,
                        timestamp=pd.Timestamp.now().isoformat()
                    ))
                
                return predictions
                
            except Exception as e:
                logger.error(f"Batch prediction error: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")
        
        @self.app.get("/model/info")
        async def model_info():
            if self.model is None:
                raise HTTPException(status_code=503, detail="Model not loaded")
            
            return {
                "model_name": self.model_name_loaded,
                "model_type": type(self.model).__name__,
                "model_params": self.model.get_params(),
                "feature_importance": self._get_feature_importance()
            }
        
        @self.app.post("/model/reload")
        async def reload_model():
            try:
                self.load_model()
                return {"message": "Model reloaded successfully", "model_name": self.model_name_loaded}
            except Exception as e:
                logger.error(f"Model reload error: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Model reload failed: {str(e)}")
    
    def load_model(self):
        logger.info("Loading model")
        
        model_files = list(self.model_path.glob(f"*.{self.save_format}"))
        
        if not model_files:
            raise FileNotFoundError(f"No model files found in {self.model_path}")
        
        latest_model_file = max(model_files, key=lambda x: x.stat().st_mtime)
        
        if self.save_format == 'pickle':
            with open(latest_model_file, 'rb') as f:
                self.model = pickle.load(f)
        else:
            import joblib
            self.model = joblib.load(latest_model_file)
        
        self.model_name_loaded = latest_model_file.stem
        
        logger.info(f"Model loaded successfully: {self.model_name_loaded}")
    
    def _get_feature_importance(self) -> Optional[Dict[str, float]]:
        if hasattr(self.model, 'feature_importances_'):
            return dict(enumerate(self.model.feature_importances_))
        elif hasattr(self.model, 'coef_'):
            return dict(enumerate(self.model.coef_[0]))
        return None
    
    def _log_prediction_metrics(self, prediction: int, latency: float):
        self.prediction_count += 1
        self.prediction_latencies.append(latency)
        
        if len(self.prediction_latencies) > 1000:
            self.prediction_latencies = self.prediction_latencies[-1000:]
    
    def run(self, host: str = None, port: int = None, workers: int = None):
        if host is None:
            host = config.get("deployment.host", "0.0.0.0")
        if port is None:
            port = config.get("deployment.port", 8000)
        if workers is None:
            workers = config.get("deployment.workers", 4)
        
        import uvicorn
        
        logger.info(f"Starting model server - host: {host}, port: {port}, workers: {workers}")
        
        uvicorn.run(
            self.app,
            host=host,
            port=port,
            workers=workers,
            log_level=config.get("logging.level", "info").lower()
        )

if __name__ == "__main__":
    server = ModelServer()
    server.run() 