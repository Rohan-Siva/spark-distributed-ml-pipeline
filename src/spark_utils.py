import yaml
from pyspark.sql import SparkSession
import os

def load_config(config_path="config/spark_config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def get_spark_session(app_name=None, config_path="config/spark_config.yaml"):

    config = load_config(config_path)
    spark_conf = config.get("spark", {})
    
    name = app_name if app_name else spark_conf.get("app_name", "DefaultSparkApp")
    master = spark_conf.get("master", "local[*]")
    
    builder = SparkSession.builder \
        .appName(name) \
        .master(master) \
        .config("spark.executor.memory", spark_conf.get("executor_memory", "2g")) \
        .config("spark.driver.memory", spark_conf.get("driver_memory", "2g"))
        
    
    spark = builder.getOrCreate()
    
    log_level = spark_conf.get("log_level", "INFO")
    spark.sparkContext.setLogLevel(log_level)
    
    return spark
