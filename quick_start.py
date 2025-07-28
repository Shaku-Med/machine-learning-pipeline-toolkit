#!/usr/bin/env python3

"""
Quick Start Guide
=================

This script helps you get started with the ML Pipeline System.
"""

import subprocess
import sys
from pathlib import Path

def run_command(command, description):
    """Run a command and show output"""
    print(f"\n🚀 {description}")
    print(f"   Running: {command}")
    print("-" * 50)
    
    try:
        result = subprocess.run(
            command.split(),
            text=True,
            timeout=120
        )
        
        if result.returncode == 0:
            print("   ✅ SUCCESS!")
        else:
            print(f"   ❌ FAILED: {result.stderr}")
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"   ❌ ERROR: {str(e)}")
        return False

def main():
    print("🎯 ML Pipeline System - Quick Start Guide")
    print("=" * 50)
    
    print("\n📋 This guide will help you:")
    print("1. Test that everything is working")
    print("2. Create sample data")
    print("3. Run the ML pipeline")
    print("4. Explore the results")
    
    print("\n🎯 Step 1: Test Imports")
    print("-" * 30)
    success = run_command("python test_imports.py", "Testing all imports")
    
    if not success:
        print("\n❌ Import test failed. Please check your installation.")
        return
    
    print("\n🎯 Step 2: Create Sample Data and Run Pipeline")
    print("-" * 50)
    success = run_command("python cli.py run-pipeline --create-sample", "Creating sample data and running full pipeline")
    
    if not success:
        print("\n❌ Pipeline failed. Let's try the core test instead.")
        run_command("python test_pipeline.py", "Testing core pipeline functionality")
    
    print("\n🎯 Step 3: Check Results")
    print("-" * 30)
    
    directories = [
        ("data/", "Processed data files"),
        ("models/", "Trained model files"),
        ("logs/", "Execution logs"),
        ("plots/", "Generated visualizations")
    ]
    
    for dir_path, description in directories:
        path = Path(dir_path)
        if path.exists():
            files = list(path.glob("*"))
            print(f"   📁 {dir_path}: {len(files)} files ({description})")
            for file in files[:3]:  # Show first 3 files
                print(f"      - {file.name}")
            if len(files) > 3:
                print(f"      ... and {len(files) - 3} more")
        else:
            print(f"   📁 {dir_path}: Not found")
    
    print("\n🎯 Step 4: Try Individual Commands")
    print("-" * 40)
    
    commands = [
        ("python cli.py status", "Check system status"),
        ("python cli.py train-models", "Train models only"),
        ("python cli.py visualize", "Generate visualizations"),
        ("python cli.py --help", "See all available commands")
    ]
    
    for command, description in commands:
        print(f"   💡 {description}: {command}")
    
    print("\n🎯 Step 5: Next Steps")
    print("-" * 30)
    print("   📚 Read the README.md for detailed documentation")
    print("   🧪 Run 'python run_all_tests.py' for comprehensive testing")
    print("   🔧 Run 'python cli.py --help' to see all available commands")
    print("   📊 Check the generated files in data/, models/, logs/, and plots/")
    
    print("\n🎉 Quick start completed!")
    print("=" * 50)

if __name__ == "__main__":
    main() 