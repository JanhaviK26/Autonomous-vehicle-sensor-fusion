"""
MLflow Configuration and Utilities

This module provides MLflow setup and utilities for experiment tracking,
model registry, and deployment management.
"""

import mlflow
import mlflow.pytorch
import mlflow.sklearn
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
import yaml
import json
from datetime import datetime


class MLflowManager:
    """Manager for MLflow experiments and model registry"""
    
    def __init__(self, tracking_uri: str = "sqlite:///mlflow.db", 
                 registry_uri: str = "sqlite:///mlflow.db"):
        self.tracking_uri = tracking_uri
        self.registry_uri = registry_uri
        
        # Set MLflow tracking URI
        mlflow.set_tracking_uri(tracking_uri)
        
        # Create experiments directory
        self.experiments_dir = Path("experiments")
        self.experiments_dir.mkdir(exist_ok=True)
        
        # Create metrics and plots directories
        self.metrics_dir = Path("metrics")
        self.plots_dir = Path("plots")
        self.metrics_dir.mkdir(exist_ok=True)
        self.plots_dir.mkdir(exist_ok=True)
    
    def setup_experiments(self):
        """Setup MLflow experiments"""
        experiments = {
            "depth_prediction": "Depth prediction using fused LiDAR and camera data",
            "segmentation": "Drivable area segmentation using fused sensor data",
            "model_comparison": "Comparison between different model architectures",
            "ablation_study": "Ablation study on sensor fusion components"
        }
        
        for exp_name, description in experiments.items():
            try:
                mlflow.create_experiment(exp_name, description)
                print(f"Created experiment: {exp_name}")
            except mlflow.exceptions.MlflowException:
                print(f"Experiment {exp_name} already exists")
    
    def log_preprocessing_metrics(self, metrics: Dict[str, Any], run_name: str = "preprocessing"):
        """Log preprocessing metrics"""
        with mlflow.start_run(run_name=run_name, experiment_id=self._get_experiment_id("depth_prediction")):
            # Log parameters
            mlflow.log_params({
                "preprocessing_method": "lidar_to_depth_conversion",
                "image_size": "256x256",
                "depth_max": 80.0
            })
            
            # Log metrics
            mlflow.log_metrics(metrics)
            
            # Save metrics to file
            self._save_metrics(metrics, "preprocessing.json")
    
    def log_training_metrics(self, metrics: Dict[str, Any], config: Dict[str, Any], 
                           model_type: str, run_name: Optional[str] = None):
        """Log training metrics"""
        if run_name is None:
            run_name = f"{model_type}_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        experiment_id = self._get_experiment_id(model_type)
        
        with mlflow.start_run(run_name=run_name, experiment_id=experiment_id):
            # Log hyperparameters
            mlflow.log_params({
                "model_name": config['model']['name'],
                "batch_size": config['training']['batch_size'],
                "learning_rate": config['training']['learning_rate'],
                "epochs": config['training']['epochs'],
                "loss_type": config['model']['loss']['type'],
                "optimizer": "Adam",
                "scheduler": config['training']['scheduler']
            })
            
            # Log metrics
            mlflow.log_metrics(metrics)
            
            # Log model
            model_path = f"models/{model_type}/best_checkpoint.pth"
            if os.path.exists(model_path):
                mlflow.pytorch.log_model(
                    pytorch_model=model_path,
                    artifact_path="model",
                    registered_model_name=f"{model_type}_model"
                )
            
            # Save metrics to file
            self._save_metrics(metrics, f"{model_type}_training.json")
    
    def log_evaluation_metrics(self, metrics: Dict[str, Any], model_type: str, 
                             run_name: Optional[str] = None):
        """Log evaluation metrics"""
        if run_name is None:
            run_name = f"{model_type}_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        experiment_id = self._get_experiment_id(model_type)
        
        with mlflow.start_run(run_name=run_name, experiment_id=experiment_id):
            # Log metrics
            mlflow.log_metrics(metrics)
            
            # Save metrics to file
            self._save_metrics(metrics, f"{model_type}_evaluation.json")
    
    def log_model_comparison(self, comparison_results: Dict[str, Dict[str, float]]):
        """Log model comparison results"""
        run_name = f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        with mlflow.start_run(run_name=run_name, experiment_id=self._get_experiment_id("model_comparison")):
            # Log comparison metrics for each model
            for model_name, metrics in comparison_results.items():
                for metric_name, value in metrics.items():
                    mlflow.log_metric(f"{model_name}_{metric_name}", value)
            
            # Save comparison results
            self._save_metrics(comparison_results, "model_comparison.json")
    
    def _get_experiment_id(self, experiment_name: str) -> str:
        """Get experiment ID by name"""
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            return experiment.experiment_id
        except mlflow.exceptions.MlflowException:
            # Create experiment if it doesn't exist
            experiment_id = mlflow.create_experiment(experiment_name)
            return experiment_id
    
    def _save_metrics(self, metrics: Dict[str, Any], filename: str):
        """Save metrics to JSON file"""
        metrics_path = self.metrics_dir / filename
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
    
    def generate_experiment_report(self) -> str:
        """Generate comprehensive experiment report"""
        report_path = Path("reports")
        report_path.mkdir(exist_ok=True)
        
        report_file = report_path / f"experiment_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        
        # Get all experiments
        experiments = mlflow.search_experiments()
        
        html_content = self._generate_html_report(experiments)
        
        with open(report_file, 'w') as f:
            f.write(html_content)
        
        print(f"Experiment report generated: {report_file}")
        return str(report_file)
    
    def _generate_html_report(self, experiments) -> str:
        """Generate HTML report content"""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Sensor Fusion Experiment Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .header { background-color: #f0f0f0; padding: 20px; border-radius: 5px; }
                .experiment { margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
                .metric { display: inline-block; margin: 5px; padding: 5px 10px; background-color: #e8f4f8; border-radius: 3px; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Sensor Fusion Experiment Report</h1>
                <p>Generated on: {}</p>
            </div>
        """.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        
        # Add experiment summaries
        for exp in experiments:
            html += f"""
            <div class="experiment">
                <h2>{exp.name}</h2>
                <p>{exp.description or 'No description available'}</p>
                <p>Runs: {exp.lifecycle_stage}</p>
            </div>
            """
        
        html += """
        </body>
        </html>
        """
        
        return html


class ModelRegistry:
    """Model registry for managing model versions and deployments"""
    
    def __init__(self):
        self.registry_uri = "sqlite:///mlflow.db"
        mlflow.set_registry_uri(self.registry_uri)
    
    def register_model(self, model_name: str, model_path: str, 
                     metrics: Dict[str, float], tags: Optional[Dict[str, str]] = None):
        """Register a model in the registry"""
        try:
            # Load the model
            model = mlflow.pytorch.load_model(model_path)
            
            # Register with metrics and tags
            with mlflow.start_run():
                mlflow.pytorch.log_model(
                    pytorch_model=model,
                    artifact_path="model",
                    registered_model_name=model_name
                )
                
                # Log metrics
                mlflow.log_metrics(metrics)
                
                # Log tags
                if tags:
                    mlflow.set_tags(tags)
            
            print(f"Model {model_name} registered successfully")
            
        except Exception as e:
            print(f"Error registering model {model_name}: {e}")
    
    def get_best_model(self, model_name: str, metric_name: str = "rmse") -> Optional[str]:
        """Get the best model based on a metric"""
        try:
            # Search for runs with the model
            runs = mlflow.search_runs(
                filter_string=f"tags.mlflow.runName LIKE '%{model_name}%'",
                order_by=[f"metrics.{metric_name} ASC"]
            )
            
            if not runs.empty:
                best_run_id = runs.iloc[0]['run_id']
                model_uri = f"runs:/{best_run_id}/model"
                return model_uri
            
        except Exception as e:
            print(f"Error getting best model: {e}")
        
        return None
    
    def list_models(self) -> List[str]:
        """List all registered models"""
        try:
            client = mlflow.tracking.MlflowClient()
            models = client.search_registered_models()
            return [model.name for model in models]
        except Exception as e:
            print(f"Error listing models: {e}")
            return []


def setup_mlflow():
    """Setup MLflow for the project"""
    manager = MLflowManager()
    manager.setup_experiments()
    
    print("MLflow setup complete!")
    print(f"Tracking URI: {manager.tracking_uri}")
    print(f"Registry URI: {manager.registry_uri}")
    
    return manager


def main():
    """Setup MLflow and generate initial report"""
    # Setup MLflow
    manager = setup_mlflow()
    
    # Generate initial report
    report_path = manager.generate_experiment_report()
    
    print(f"\nMLflow setup complete!")
    print(f"Experiment report: {report_path}")
    print(f"MLflow UI: mlflow ui --backend-store-uri sqlite:///mlflow.db")


if __name__ == "__main__":
    main()
