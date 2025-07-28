#!/usr/bin/env python3

"""
Core ML Pipeline Test
=====================

This script tests the core functionality without optional dependencies.
"""

import sys
import traceback
from pathlib import Path

def test_imports():
    """Test that core modules can be imported"""
    print("Testing imports...")
    
    try:
        import numpy as np
        print("✅ numpy imported successfully")
    except ImportError as e:
        print(f"❌ numpy import failed: {e}")
        return False
    
    try:
        import pandas as pd
        print("✅ pandas imported successfully")
    except ImportError as e:
        print(f"❌ pandas import failed: {e}")
        return False
    
    try:
        import sklearn
        print("✅ scikit-learn imported successfully")
    except ImportError as e:
        print(f"❌ scikit-learn import failed: {e}")
        return False
    
    try:
        from src.utils.config import config
        print("✅ config module imported successfully")
    except ImportError as e:
        print(f"❌ config module import failed: {e}")
        return False
    
    try:
        from src.utils.logger_simple import logger
        print("✅ logger module imported successfully")
    except ImportError as e:
        print(f"❌ logger module import failed: {e}")
        return False
    
    return True

def test_data_pipeline():
    """Test data pipeline functionality"""
    print("\nTesting data pipeline...")
    
    try:
        from src.data.pipeline import DataPipeline
        print("✅ DataPipeline imported successfully")
        
        pipeline = DataPipeline()
        print("✅ DataPipeline instantiated successfully")
        
        return True
    except Exception as e:
        print(f"❌ Data pipeline test failed: {e}")
        traceback.print_exc()
        return False

def test_model_training():
    """Test model training functionality"""
    print("\nTesting model training...")
    
    try:
        from src.models.train_simple import ModelTrainer
        print("✅ ModelTrainer imported successfully")
        
        trainer = ModelTrainer()
        print("✅ ModelTrainer instantiated successfully")
        
        return True
    except Exception as e:
        print(f"❌ Model training test failed: {e}")
        traceback.print_exc()
        return False

def test_config():
    """Test configuration loading"""
    print("\nTesting configuration...")
    
    try:
        from src.utils.config import config
        
        app_name = config.get("app.name")
        print(f"✅ Configuration loaded: {app_name}")
        
        data_path = config.get("data.raw_data_path")
        print(f"✅ Data path configured: {data_path}")
        
        return True
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        traceback.print_exc()
        return False

def test_sample_data_creation():
    """Test sample data creation"""
    print("\nTesting sample data creation...")
    
    try:
        import numpy as np
        import pandas as pd
        from pathlib import Path
        
        np.random.seed(42)
        sample_data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'feature2': np.random.normal(0, 1, 100),
            'target': np.random.choice([0, 1], 100)
        })
        
        data_path = Path("data/raw")
        data_path.mkdir(parents=True, exist_ok=True)
        
        sample_data.to_csv(data_path / "test_sample.csv", index=False)
        print("✅ Sample data created successfully")
        
        return True
    except Exception as e:
        print(f"❌ Sample data creation failed: {e}")
        traceback.print_exc()
        return False

def main():
    print("🧪 Core ML Pipeline Test")
    print("=" * 40)
    
    tests = [
        ("Imports", test_imports),
        ("Configuration", test_config),
        ("Sample Data Creation", test_sample_data_creation),
        ("Data Pipeline", test_data_pipeline),
        ("Model Training", test_model_training),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} failed")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The core ML pipeline is working.")
        print("\nYou can now run:")
        print("  python demo_simple.py")
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
        print("\nTry installing missing dependencies:")
        print("  pip install -r requirements-minimal.txt")

if __name__ == "__main__":
    main() 