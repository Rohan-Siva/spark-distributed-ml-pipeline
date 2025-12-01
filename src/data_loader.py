from pyspark.sql import SparkSession
from pyspark.sql.functions import rand, randn, col
from src.spark_utils import get_spark_session

class DataLoader:
    def __init__(self, spark: SparkSession = None):
        self.spark = spark if spark else get_spark_session(app_name="DataLoader")

    def load_data(self, file_path, format="csv", **options):

        reader = self.spark.read.format(format)
        for key, value in options.items():
            reader = reader.option(key, value)
        return reader.load(file_path)

    def generate_synthetic_data(self, num_rows=100000, num_features=10, output_path=None):
 
        df = self.spark.range(0, num_rows) # feature gen
        
        for i in range(num_features):
            df = df.withColumn(f"feature_{i}", randn(seed=i))
            
        #Binary classification target gen
        df = df.withColumn("target_raw", 
                           col("feature_0") * 2 + col("feature_1") * -1.5 + randn())
        df = df.withColumn("label", (col("target_raw") > 0).cast("integer"))
        df = df.drop("target_raw")
        
        if output_path:
            df.write.mode("overwrite").parquet(output_path)
            print(f"Synthetic data saved to {output_path}")
            
        return df

if __name__ == "__main__":
    loader = DataLoader()
    df = loader.generate_synthetic_data(num_rows=1000, num_features=5)
    df.show(5)
