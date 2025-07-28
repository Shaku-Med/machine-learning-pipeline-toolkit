#!/usr/bin/env python3

"""
ML Pipeline System Demo (Simplified)
====================================

This script demonstrates the ML pipeline system without optional dependencies.
"""

import pandas as pd
import numpy as np
import time
from pathlib import Path
from src.pipeline_simple import MLPipelineOrchestrator, create_sample_data
from src.utils.config import config
from src.utils.logger_simple import logger

def main():
    print("🚀 ML Pipeline System Demo (Simplified)")
    print("=" * 50)
    
    orchestrator = MLPipelineOrchestrator()
    
    print("\n1️⃣ Creating sample data...")
    sample_data_path = create_sample_data()
    print(f"   Sample data created: {sample_data_path}")
    
    print("\n2️⃣ Running complete ML pipeline...")
    start_time = time.time()
    
    try:
        results = orchestrator.run_full_pipeline(
            data_path=sample_data_path,
            target_column="target"
        )
        
        pipeline_time = time.time() - start_time
        print(f"   ✅ Pipeline completed in {pipeline_time:.2f} seconds")
        
        best_model = results['training']['best_model_name']
        test_accuracy = results['training']['final_evaluation']['test_metrics']['accuracy']
        
        print(f"   Best model: {best_model}")
        print(f"   Test accuracy: {test_accuracy:.4f}")
        
    except Exception as e:
        print(f"   ❌ Pipeline failed: {str(e)}")
        return
    
    print("\n3️⃣ Testing data pipeline only...")
    try:
        data_results = orchestrator.run_data_only(sample_data_path, "target")
        print(f"   ✅ Data pipeline completed")
        print(f"   Features: {len(data_results['data_pipeline']['feature_columns'])}")
        
    except Exception as e:
        print(f"   ⚠️ Data pipeline test failed: {str(e)}")
    
    print("\n4️⃣ Testing model training only...")
    try:
        training_results = orchestrator.run_training_only()
        print(f"   ✅ Model training completed")
        print(f"   Best model: {training_results['best_model_name']}")
        
    except Exception as e:
        print(f"   ⚠️ Model training test failed: {str(e)}")
    
    print("\n5️⃣ Testing model deployment...")
    try:
        server = orchestrator.deploy_model_only()
        print(f"   ✅ Model server created successfully")
        print(f"   Model loaded: {server.model_name_loaded}")
        
    except Exception as e:
        print(f"   ⚠️ Model deployment test failed: {str(e)}")
        print("   (This is expected if FastAPI/uvicorn is not installed)")
    
    print("\n📊 Demo Summary")
    print("=" * 50)
    print(f"Pipeline execution time: {pipeline_time:.2f} seconds")
    print(f"Best model: {best_model}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Features processed: {len(results['data_pipeline']['feature_columns'])}")
    print(f"Models trained: {len(results['training']['all_results'])}")
    
    print("\n🎯 Next Steps:")
    print("1. Check 'data/processed' directory for processed data")
    print("2. Check 'models' directory for trained models")
    print("3. Check 'logs' directory for execution logs")
    print("4. Run 'python cli.py --help' for more commands")
    
    print("\n✨ Demo completed successfully!")

if __name__ == "__main__":
    main() 