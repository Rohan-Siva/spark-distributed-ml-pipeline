from pyspark.sql import SparkSession
import os
import sys

print(f"Python: {sys.version}")
print(f"JAVA_HOME: {os.environ.get('JAVA_HOME')}")

try:
    spark = SparkSession.builder.master("local").appName("Test").getOrCreate()
    print("Spark Session created successfully")
    df = spark.range(10)
    print(f"Count: {df.count()}")
    spark.stop()
except Exception as e:
    print(f"Error: {e}")
