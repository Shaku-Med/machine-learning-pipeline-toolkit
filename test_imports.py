#!/usr/bin/env python3

"""
Import Test Script
==================

This script tests that all the simplified modules can be imported correctly.
"""

def test_imports():
    """Test all the simplified imports"""
    print("🧪 Testing imports...")
    
    try:
        print("   Testing config...")
        from src.utils.config import config
        print("   ✅ Config imported successfully")
        
        print("   Testing logger...")
        from src.utils.logger_simple import logger
        print("   ✅ Logger imported successfully")
        
        print("   Testing data pipeline...")
        from src.data.pipeline import DataPipeline
        print("   ✅ Data pipeline imported successfully")
        
        print("   Testing model training...")
        from src.models.train_simple import ModelTrainer
        print("   ✅ Model trainer imported successfully")
        
        print("   Testing model deployment...")
        from src.deployment.server_simple import ModelServer
        print("   ✅ Model server imported successfully")
        
        print("   Testing drift detection...")
        from src.monitoring.drift_detector_simple import DriftDetector
        print("   ✅ Drift detector imported successfully")
        
        print("   Testing visualization...")
        from src.utils.visualization_simple import ModelVisualizer
        print("   ✅ Visualization imported successfully")
        
        print("   Testing pipeline orchestrator...")
        from src.pipeline_simple import MLPipelineOrchestrator
        print("   ✅ Pipeline orchestrator imported successfully")
        
        print("   Testing CLI...")
        import cli
        print("   ✅ CLI imported successfully")
        
        print("\n🎉 All imports successful!")
        return True
        
    except Exception as e:
        print(f"   ❌ Import failed: {str(e)}")
        return False

def test_basic_functionality():
    """Test basic functionality of key components"""
    print("\n🧪 Testing basic functionality...")
    
    try:
        from src.utils.config import config
        from src.utils.logger_simple import logger
        from src.pipeline_simple import create_sample_data
        
        print("   Testing sample data creation...")
        data_path = create_sample_data()
        print(f"   ✅ Sample data created: {data_path}")
        
        print("   Testing logger...")
        logger.info("Test log message")
        print("   ✅ Logger working")
        
        print("   Testing config...")
        print(f"   ✅ Config loaded: {len(config)} sections")
        
        print("\n🎉 Basic functionality working!")
        return True
        
    except Exception as e:
        print(f"   ❌ Basic functionality failed: {str(e)}")
        return False

def main():
    print("🚀 Import and Functionality Test")
    print("=" * 40)
    
    import_success = test_imports()
    functionality_success = test_basic_functionality()
    
    print(f"\n📊 Test Results")
    print("=" * 40)
    print(f"✅ Imports: {'PASS' if import_success else 'FAIL'}")
    print(f"✅ Functionality: {'PASS' if functionality_success else 'FAIL'}")
    
    if import_success and functionality_success:
        print("\n🎉 All tests passed! The ML pipeline system is ready to use.")
        print("\n🎯 Next Steps:")
        print("1. Run 'python test_pipeline.py' for core functionality")
        print("2. Run 'python test_cli.py' for CLI testing")
        print("3. Run 'python demo_simple.py' for full demo")
    else:
        print("\n⚠️ Some tests failed. Check the error messages above.")

if __name__ == "__main__":
    main() 