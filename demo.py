#!/usr/bin/env python3

"""
ML Pipeline System Demo
=======================

This script demonstrates the complete ML pipeline system with:
1. Data generation and preprocessing
2. Model training with multiple algorithms
3. Model evaluation and comparison
4. Model deployment and serving
5. Monitoring and drift detection
6. Visualization generation
"""

import pandas as pd
import numpy as np
import time
import requests
import json
from pathlib import Path
from src.pipeline import MLPipelineOrchestrator, create_sample_data
from src.utils.visualization import ModelVisualizer
from src.utils.config import config
from src.utils.logger import logger

def main():
    print("🚀 ML Pipeline System Demo")
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
    
    print("\n3️⃣ Generating visualizations...")
    try:
        visualizer = ModelVisualizer("demo_plots")
        
        training_history = results['training']['training_history']
        visualizer.plot_training_metrics(training_history)
        
        if 'feature_importance' in results['training']['final_evaluation']:
            feature_importance = results['training']['final_evaluation']['feature_importance']
            if feature_importance:
                visualizer.plot_feature_importance(feature_importance)
        
        print("   ✅ Visualizations created in 'demo_plots' directory")
        
    except Exception as e:
        print(f"   ⚠️ Visualization failed: {str(e)}")
    
    print("\n4️⃣ Testing model serving...")
    try:
        server = orchestrator.deploy_model_only()
        print("   ✅ Model server created")
        
        print("   Testing prediction endpoint...")
        
        test_features = [0.5, 0.3, 1, 0.8, 5.0]
        
        prediction_data = {
            "features": test_features,
            "feature_names": ["feature1", "feature2", "feature3", "feature4", "feature5"]
        }
        
        response = requests.post(
            "http://localhost:8000/predict",
            json=prediction_data,
            timeout=10
        )
        
        if response.status_code == 200:
            prediction = response.json()
            print(f"   ✅ Prediction successful: {prediction}")
        else:
            print(f"   ⚠️ Prediction failed: {response.status_code}")
            
    except Exception as e:
        print(f"   ⚠️ Model serving test failed: {str(e)}")
    
    print("\n5️⃣ Testing monitoring...")
    try:
        reference_data_path = Path(config.get("data.processed_data_path")) / "train.csv"
        monitor = orchestrator.setup_monitoring_only(str(reference_data_path))
        
        print("   ✅ Monitoring setup completed")
        
        summary = monitor.get_monitoring_summary()
        print(f"   Monitoring summary: {summary['total_predictions']} predictions tracked")
        
    except Exception as e:
        print(f"   ⚠️ Monitoring test failed: {str(e)}")
    
    print("\n6️⃣ Creating interactive dashboard...")
    try:
        dashboard_path = visualizer.create_interactive_dashboard(results['training'])
        print(f"   ✅ Interactive dashboard created: {dashboard_path}")
        
    except Exception as e:
        print(f"   ⚠️ Dashboard creation failed: {str(e)}")
    
    print("\n📊 Demo Summary")
    print("=" * 50)
    print(f"Pipeline execution time: {pipeline_time:.2f} seconds")
    print(f"Best model: {best_model}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Features processed: {len(results['data_pipeline']['feature_columns'])}")
    print(f"Models trained: {len(results['training']['all_results'])}")
    
    print("\n🎯 Next Steps:")
    print("1. Visit http://localhost:8000/docs for API documentation")
    print("2. Check 'demo_plots' directory for visualizations")
    print("3. Run 'python cli.py --help' for more commands")
    print("4. Explore the codebase in the 'src' directory")
    
    print("\n✨ Demo completed successfully!")

if __name__ == "__main__":
    main() 