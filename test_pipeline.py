#!/usr/bin/env python3

"""
Simple ML Pipeline Test
=======================

This script tests the core ML pipeline functionality without deployment.
"""

import time
from src.pipeline_simple import MLPipelineOrchestrator, create_sample_data
from src.utils.logger_simple import logger

def main():
    print("🧪 Simple ML Pipeline Test")
    print("=" * 40)
    
    orchestrator = MLPipelineOrchestrator()
    
    print("\n1️⃣ Creating sample data...")
    sample_data_path = create_sample_data()
    print(f"   ✅ Sample data created: {sample_data_path}")
    
    print("\n2️⃣ Running data pipeline only...")
    try:
        data_results = orchestrator.run_data_only(sample_data_path, "target")
        print(f"   ✅ Data pipeline completed")
        print(f"   Features: {len(data_results['data_pipeline']['feature_columns'])}")
        print(f"   Data shapes: {data_results['data_pipeline']['data_shapes']}")
        
    except Exception as e:
        print(f"   ❌ Data pipeline failed: {str(e)}")
        return
    
    print("\n3️⃣ Running model training only...")
    start_time = time.time()
    
    try:
        training_results = orchestrator.run_training_only()
        training_time = time.time() - start_time
        
        print(f"   ✅ Model training completed in {training_time:.2f} seconds")
        print(f"   Best model: {training_results['best_model_name']}")
        print(f"   Test accuracy: {training_results['final_evaluation']['test_metrics']['accuracy']:.4f}")
        print(f"   Models trained: {len(training_results['all_results'])}")
        
    except Exception as e:
        print(f"   ❌ Model training failed: {str(e)}")
        return
    
    print("\n📊 Test Results")
    print("=" * 40)
    print(f"✅ Data pipeline: SUCCESS")
    print(f"✅ Model training: SUCCESS")
    print(f"✅ Best model: {training_results['best_model_name']}")
    print(f"✅ Test accuracy: {training_results['final_evaluation']['test_metrics']['accuracy']:.4f}")
    print(f"✅ Training time: {training_time:.2f} seconds")
    
    print("\n🎯 Next Steps:")
    print("1. Check 'data/processed' directory for processed data")
    print("2. Check 'models' directory for trained models")
    print("3. Run 'python demo_simple.py' for full demo")
    
    print("\n✨ Test completed successfully!")

if __name__ == "__main__":
    main() 