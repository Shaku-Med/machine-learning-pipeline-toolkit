#!/usr/bin/env python3

"""
CLI Test Script
===============

This script tests the CLI functionality.
"""

import subprocess
import sys
from pathlib import Path

def test_cli_help():
    """Test that CLI help works"""
    print("🧪 Testing CLI help...")
    
    try:
        result = subprocess.run(
            [sys.executable, "cli.py", "--help"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            print("   ✅ CLI help works")
            return True
        else:
            print(f"   ❌ CLI help failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"   ❌ CLI help failed: {str(e)}")
        return False

def test_cli_commands():
    """Test that CLI commands are available"""
    print("\n🧪 Testing CLI commands...")
    
    try:
        result = subprocess.run(
            [sys.executable, "cli.py", "--help"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            output = result.stdout.lower()
            
            commands = [
                "run-pipeline",
                "data-pipeline", 
                "train-models",
                "serve-model",
                "setup-monitoring",
                "visualize",
                "status",
                "test"
            ]
            
            missing_commands = []
            for cmd in commands:
                if cmd not in output:
                    missing_commands.append(cmd)
            
            if not missing_commands:
                print("   ✅ All CLI commands available")
                return True
            else:
                print(f"   ⚠️ Missing commands: {missing_commands}")
                return False
                
        else:
            print(f"   ❌ CLI commands test failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"   ❌ CLI commands test failed: {str(e)}")
        return False

def test_sample_data_creation():
    """Test sample data creation via CLI"""
    print("\n🧪 Testing sample data creation...")
    
    try:
        result = subprocess.run(
            [sys.executable, "cli.py", "run-pipeline", "--create-sample"],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            print("   ✅ Sample data creation works")
            return True
        else:
            print(f"   ⚠️ Sample data creation failed: {result.stderr}")
            print("   (This might be expected if dependencies are missing)")
            return False
            
    except Exception as e:
        print(f"   ⚠️ Sample data creation failed: {str(e)}")
        print("   (This might be expected if dependencies are missing)")
        return False

def main():
    print("🚀 CLI Test Suite")
    print("=" * 40)
    
    tests = [
        test_cli_help,
        test_cli_commands,
        test_sample_data_creation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n📊 Test Results")
    print("=" * 40)
    print(f"✅ Passed: {passed}/{total}")
    print(f"❌ Failed: {total - passed}/{total}")
    
    if passed == total:
        print("\n🎉 All CLI tests passed!")
    else:
        print("\n⚠️ Some CLI tests failed (this might be expected)")
    
    print("\n🎯 Next Steps:")
    print("1. Run 'python cli.py --help' to see all commands")
    print("2. Run 'python cli.py run-pipeline --create-sample' to test full pipeline")
    print("3. Run 'python test_pipeline.py' for core functionality test")

if __name__ == "__main__":
    main() 