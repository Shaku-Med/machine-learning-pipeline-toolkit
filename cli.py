#!/usr/bin/env python3

import click
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
from src.pipeline_simple import MLPipelineOrchestrator, create_sample_data
from src.utils.config import config
from src.utils.logger_simple import logger
from src.utils.visualization_simple import ModelVisualizer

@click.group()
@click.version_option(version="1.0.0")
def cli():
    """ML Pipeline System - A comprehensive ML engineering platform"""
    pass

@cli.command()
@click.option('--data-path', '-d', help='Path to input data file')
@click.option('--target-column', '-t', help='Name of target column')
@click.option('--create-sample', is_flag=True, help='Create sample data for testing')
def run_pipeline(data_path: Optional[str], target_column: Optional[str], create_sample: bool):
    """Run the complete ML pipeline"""
    
    if create_sample:
        logger.info("Creating sample data")
        data_path = create_sample_data()
        target_column = "target"
    elif not data_path or not target_column:
        click.echo("Error: Please provide both --data-path and --target-column, or use --create-sample")
        return
    
    click.echo(f"Starting ML pipeline with data: {data_path}")
    click.echo(f"Target column: {target_column}")
    
    orchestrator = MLPipelineOrchestrator()
    
    try:
        results = orchestrator.run_full_pipeline(data_path, target_column)
        
        click.echo("✅ Pipeline completed successfully!")
        click.echo(f"Best model: {results['training']['best_model_name']}")
        click.echo(f"Test accuracy: {results['training']['final_evaluation']['test_metrics']['accuracy']:.4f}")
        
    except Exception as e:
        click.echo(f"❌ Pipeline failed: {str(e)}")
        logger.error(f"Pipeline execution failed: {str(e)}")

@cli.command()
@click.option('--data-path', '-d', required=True, help='Path to input data file')
@click.option('--target-column', '-t', required=True, help='Name of target column')
def data_pipeline(data_path: str, target_column: str):
    """Run only the data processing pipeline"""
    
    click.echo(f"Running data pipeline for: {data_path}")
    
    orchestrator = MLPipelineOrchestrator()
    
    try:
        results = orchestrator.run_data_only(data_path, target_column)
        click.echo("✅ Data pipeline completed successfully!")
        click.echo(f"Features: {len(results['data_pipeline']['feature_columns'])}")
        click.echo(f"Data shapes: {results['data_pipeline']['data_shapes']}")
        
    except Exception as e:
        click.echo(f"❌ Data pipeline failed: {str(e)}")
        logger.error(f"Data pipeline failed: {str(e)}")

@cli.command()
def train_models():
    """Run only the model training pipeline"""
    
    click.echo("Running model training pipeline")
    
    orchestrator = MLPipelineOrchestrator()
    
    try:
        results = orchestrator.run_training_only()
        click.echo("✅ Model training completed successfully!")
        click.echo(f"Best model: {results['best_model_name']}")
        click.echo(f"Test accuracy: {results['final_evaluation']['test_metrics']['accuracy']:.4f}")
        
    except Exception as e:
        click.echo(f"❌ Model training failed: {str(e)}")
        logger.error(f"Model training failed: {str(e)}")

@cli.command()
@click.option('--host', default=None, help='Server host')
@click.option('--port', default=None, type=int, help='Server port')
def serve_model(host: Optional[str], port: Optional[int]):
    """Start the model serving server"""
    
    click.echo("Starting model server...")
    
    orchestrator = MLPipelineOrchestrator()
    
    try:
        server = orchestrator.deploy_model_only()
        click.echo("✅ Model server started successfully!")
        click.echo(f"API docs: http://{host or 'localhost'}:{port or 8000}/docs")
        click.echo(f"Health check: http://{host or 'localhost'}:{port or 8000}/health")
        
        server.run(host=host, port=port)
        
    except Exception as e:
        click.echo(f"❌ Model server failed: {str(e)}")
        logger.error(f"Model server failed: {str(e)}")

@cli.command()
@click.option('--reference-data', '-r', required=True, help='Path to reference data file')
def setup_monitoring(reference_data: str):
    """Setup model monitoring with reference data"""
    
    click.echo(f"Setting up monitoring with reference data: {reference_data}")
    
    orchestrator = MLPipelineOrchestrator()
    
    try:
        monitor = orchestrator.setup_monitoring_only(reference_data)
        click.echo("✅ Monitoring setup completed successfully!")
        
    except Exception as e:
        click.echo(f"❌ Monitoring setup failed: {str(e)}")
        logger.error(f"Monitoring setup failed: {str(e)}")

@cli.command()
@click.option('--training-results', '-t', help='Path to training results JSON file')
@click.option('--drift-results', '-d', help='Path to drift results JSON file')
@click.option('--output-dir', '-o', default='plots', help='Output directory for plots')
def visualize(training_results: Optional[str], drift_results: Optional[str], output_dir: str):
    """Generate visualizations for model performance and drift detection"""
    
    click.echo("Generating visualizations...")
    
    visualizer = ModelVisualizer(output_dir)
    
    if training_results:
        import json
        with open(training_results, 'r') as f:
            training_data = json.load(f)
        
        visualizer.plot_training_metrics(training_data.get('training_history', {}))
        click.echo(f"✅ Training metrics plot saved to {output_dir}")
    
    if drift_results:
        import json
        with open(drift_results, 'r') as f:
            drift_data = json.load(f)
        
        visualizer.plot_drift_detection(drift_data)
        click.echo(f"✅ Drift detection plot saved to {output_dir}")
    
    click.echo("✅ Visualizations completed!")

@cli.command()
def status():
    """Show the current status of the ML pipeline system"""
    
    click.echo("ML Pipeline System Status")
    click.echo("=" * 40)
    
    config_data = config.get_all()
    
    click.echo(f"Environment: {config_data.get('app', {}).get('environment', 'unknown')}")
    click.echo(f"Data paths:")
    click.echo(f"  Raw: {config_data.get('data', {}).get('raw_data_path', 'not set')}")
    click.echo(f"  Processed: {config_data.get('data', {}).get('processed_data_path', 'not set')}")
    click.echo(f"  Models: {config_data.get('model', {}).get('model_path', 'not set')}")
    
    click.echo(f"Deployment:")
    click.echo(f"  Host: {config_data.get('deployment', {}).get('host', 'not set')}")
    click.echo(f"  Port: {config_data.get('deployment', {}).get('port', 'not set')}")
    
    click.echo(f"Monitoring:")
    click.echo(f"  Drift threshold: {config_data.get('monitoring', {}).get('drift_threshold', 'not set')}")
    click.echo(f"  Metrics port: {config_data.get('monitoring', {}).get('metrics_port', 'not set')}")

@cli.command()
def test():
    """Run the test suite"""
    
    click.echo("Running tests...")
    
    import subprocess
    import sys
    
    try:
        result = subprocess.run([sys.executable, '-m', 'pytest', 'tests/', '-v'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            click.echo("✅ All tests passed!")
        else:
            click.echo("❌ Some tests failed!")
            click.echo(result.stdout)
            click.echo(result.stderr)
            
    except Exception as e:
        click.echo(f"❌ Test execution failed: {str(e)}")

@cli.command()
@click.option('--format', '-f', default='black', type=click.Choice(['black', 'flake8', 'mypy']), 
              help='Code formatting tool')
def format_code(format: str):
    """Format and lint the code"""
    
    click.echo(f"Running {format}...")
    
    import subprocess
    import sys
    
    try:
        if format == 'black':
            result = subprocess.run([sys.executable, '-m', 'black', 'src/', 'tests/'], 
                                  capture_output=True, text=True)
        elif format == 'flake8':
            result = subprocess.run([sys.executable, '-m', 'flake8', 'src/', 'tests/'], 
                                  capture_output=True, text=True)
        elif format == 'mypy':
            result = subprocess.run([sys.executable, '-m', 'mypy', 'src/'], 
                                  capture_output=True, text=True)
        
        if result.returncode == 0:
            click.echo(f"✅ {format} completed successfully!")
        else:
            click.echo(f"⚠️ {format} found issues:")
            click.echo(result.stdout)
            click.echo(result.stderr)
            
    except Exception as e:
        click.echo(f"❌ {format} execution failed: {str(e)}")

if __name__ == '__main__':
    cli() 