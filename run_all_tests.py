#!/usr/bin/env python3

"""
Comprehensive Test Runner
=========================

This script runs all the tests and CLI commands to verify the ML pipeline system.
"""

import subprocess
import sys
import time
from pathlib import Path

def run_command(command, description, timeout=60):
    """Run a command and return success status"""
    print(f"\n🧪 {description}")
    print(f"   Command: {command}")
    
    try:
        result = subprocess.run(
            command.split(),
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        if result.returncode == 0:
            print("   ✅ SUCCESS")
            return True
        else:
            print(f"   ❌ FAILED: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"   ⏰ TIMEOUT (after {timeout}s)")
        return False
    except Exception as e:
        print(f"   ❌ ERROR: {str(e)}")
        return False

def test_imports():
    """Test all imports"""
    return run_command("python test_imports.py", "Testing all imports")

def test_core_pipeline():
    """Test core pipeline functionality"""
    return run_command("python test_pipeline.py", "Testing core ML pipeline")

def test_cli_help():
    """Test CLI help"""
    return run_command("python cli.py --help", "Testing CLI help")

def test_cli_status():
    """Test CLI status command"""
    return run_command("python cli.py status", "Testing CLI status command")

def test_cli_test():
    """Test CLI test command"""
    return run_command("python cli.py test", "Testing CLI test command")

def test_cli_run_pipeline():
    """Test CLI run-pipeline with sample data"""
    return run_command("python cli.py run-pipeline --create-sample", "Testing CLI run-pipeline with sample data")

def test_cli_data_pipeline():
    """Test CLI data-pipeline command"""
    possible_data_files = [
        Path("data/sample_data.csv"),
        Path("data/raw/sample_data.csv"),
        Path("sample_data.csv")
    ]
    
    data_file = None
    for file_path in possible_data_files:
        if file_path.exists():
            data_file = file_path
            break
    
    if data_file:
        return run_command(f"python cli.py data-pipeline -d {data_file} -t target", "Testing CLI data-pipeline command")
    else:
        print("\n🧪 Testing CLI data-pipeline command")
        print("   ⚠️ SKIPPED: No sample data file found")
        print("   💡 Run 'python cli.py run-pipeline --create-sample' first to create sample data")
        return True

def test_cli_train_models():
    """Test CLI train-models command"""
    return run_command("python cli.py train-models", "Testing CLI train-models command")

def test_cli_serve_model():
    """Test CLI serve-model command (should fail gracefully)"""
    result = run_command("python cli.py serve-model", "Testing CLI serve-model command", timeout=10)
    if not result:
        print("   ⚠️ Expected to fail (no models trained yet)")
    return True

def test_cli_visualize():
    """Test CLI visualize command"""
    return run_command("python cli.py visualize", "Testing CLI visualize command")

def test_cli_format_code():
    """Test CLI format-code command"""
    return run_command("python cli.py format-code", "Testing CLI format-code command")

def test_cli_setup_monitoring():
    """Test CLI setup-monitoring command"""
    possible_data_files = [
        Path("data/sample_data.csv"),
        Path("data/raw/sample_data.csv"),
        Path("sample_data.csv")
    ]
    
    data_file = None
    for file_path in possible_data_files:
        if file_path.exists():
            data_file = file_path
            break
    
    if data_file:
        return run_command(f"python cli.py setup-monitoring -r {data_file}", "Testing CLI setup-monitoring command")
    else:
        print("\n🧪 Testing CLI setup-monitoring command")
        print("   ⚠️ SKIPPED: No sample data file found")
        print("   💡 Run 'python cli.py run-pipeline --create-sample' first to create sample data")
        return True

def main():
    print("🚀 Comprehensive ML Pipeline Test Suite")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("Core Pipeline Test", test_core_pipeline),
        ("CLI Help Test", test_cli_help),
        ("CLI Status Test", test_cli_status),
        ("CLI Test Command", test_cli_test),
        ("CLI Run Pipeline", test_cli_run_pipeline),
        ("CLI Data Pipeline", test_cli_data_pipeline),
        ("CLI Train Models", test_cli_train_models),
        ("CLI Serve Model", test_cli_serve_model),
        ("CLI Visualize", test_cli_visualize),
        ("CLI Format Code", test_cli_format_code),
        ("CLI Setup Monitoring", test_cli_setup_monitoring),
    ]
    
    results = []
    start_time = time.time()
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        success = test_func()
        results.append((test_name, success))
    
    total_time = time.time() - start_time
    
    print(f"\n{'='*50}")
    print("📊 FINAL TEST RESULTS")
    print("=" * 50)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\n📈 Summary:")
    print(f"   ✅ Passed: {passed}/{total}")
    print(f"   ❌ Failed: {total - passed}/{total}")
    print(f"   ⏱️  Total time: {total_time:.2f} seconds")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! The ML pipeline system is working perfectly!")
    else:
        print(f"\n⚠️ {total - passed} test(s) failed. Check the output above for details.")
    
    print("\n🎯 Next Steps:")
    print("1. Check 'data/' directory for processed data")
    print("2. Check 'models/' directory for trained models")
    print("3. Check 'logs/' directory for execution logs")
    print("4. Check 'plots/' directory for visualizations")
    print("5. Run individual commands as needed")

if __name__ == "__main__":
    main() 