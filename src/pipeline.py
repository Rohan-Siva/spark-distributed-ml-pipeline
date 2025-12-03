import sys
import os
import argparse
from pyspark.sql import SparkSession

# Add current directory to path
sys.path.append(os.getcwd())

from src.spark_utils import get_spark_session
from src.data_loader import DataLoader
from src.feature_engineering import FeatureEngineer
from src.model_training import ModelTrainer
from src.evaluation import ModelEvaluator

def main(args):
    spark = get_spark_session(app_name="DistributedMLPipeline")
    
    loader = DataLoader(spark)
    if args.input_path:
        df = loader.load_data(args.input_path)
    else:
        print("No input path provided, generating synthetic data...")
        df = loader.generate_synthetic_data(num_rows=1000, num_features=10)
        
    engineer = FeatureEngineer(spark)
    
    feature_cols = [c for c in df.columns if c.startswith("feature")]
    df_assembled = engineer.assemble_features(df, input_cols=feature_cols)
    feature_cols = [c for c in df.columns if c.startswith("feature")]
    df_assembled = engineer.assemble_features(df, input_cols=feature_cols)
    
    # Split data
    train_df, test_df = df_assembled.randomSplit([0.8, 0.2], seed=42)
    
    # Train Model
    trainer = ModelTrainer(spark)
    print(f"Training {args.model_type} model...")
    
    if args.tune:
        model = trainer.tune_hyperparameters(train_df, model_type=args.model_type, experiment_name="pipeline_experiment")
    else:
        model = trainer.train_model(train_df, model_type=args.model_type, experiment_name="pipeline_experiment")
        
    # Evaluate
    evaluator = ModelEvaluator(spark)
    predictions = model.transform(test_df)
    auc = evaluator.evaluate(predictions, metric_name="areaUnderROC")
    print(f"Model AUC: {auc}")
    
    spark.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distributed ML Pipeline")
    parser.add_argument("--input_path", type=str, help="Path to input data")
    parser.add_argument("--model_type", type=str, default="logistic_regression", choices=["logistic_regression", "random_forest", "gbt"], help="Model type")
    parser.add_argument("--tune", action="store_true", help="Perform hyperparameter tuning")
    
    args = parser.parse_args()
    main(args)
