import pytest
from pyspark.sql import SparkSession
from src.spark_utils import get_spark_session
from src.data_loader import DataLoader
from src.feature_engineering import FeatureEngineer
from src.model_training import ModelTrainer

@pytest.fixture(scope="session")
def spark():
    return get_spark_session(app_name="TestApp")

def test_data_loader(spark):
    loader = DataLoader(spark)
    df = loader.generate_synthetic_data(num_rows=10, num_features=2)
    assert df.count() == 10
    assert "label" in df.columns

def test_feature_engineering(spark):
    loader = DataLoader(spark)
    df = loader.generate_synthetic_data(num_rows=10, num_features=2)
    
    engineer = FeatureEngineer(spark)
    feature_cols = ["feature_0", "feature_1"]
    df_assembled = engineer.assemble_features(df, input_cols=feature_cols)
    
    assert "features" in df_assembled.columns

def test_model_training(spark):
    loader = DataLoader(spark)
    df = loader.generate_synthetic_data(num_rows=20, num_features=2)
    
    engineer = FeatureEngineer(spark)
    feature_cols = ["feature_0", "feature_1"]
    df_assembled = engineer.assemble_features(df, input_cols=feature_cols)
    
    trainer = ModelTrainer(spark)
    model = trainer.train_model(df_assembled, model_type="logistic_regression")
    
    assert model is not None
