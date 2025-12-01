from pyspark.sql import DataFrame
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer, OneHotEncoder, Imputer
from pyspark.ml import Pipeline

class FeatureEngineer:
    def __init__(self, spark):
        self.spark = spark

    def handle_missing_values(self, df: DataFrame, input_cols: list, strategy="mean"):

        output_cols = [f"{col}_imputed" for col in input_cols]
        imputer = Imputer(inputCols=input_cols, outputCols=output_cols)
        imputer.setStrategy(strategy)
        
        model = imputer.fit(df)
        return model.transform(df)

    def encode_categorical_features(self, df: DataFrame, input_cols: list):

        stages = []
        for col in input_cols:
            indexer = StringIndexer(inputCol=col, outputCol=f"{col}_indexed", handleInvalid="keep")
            encoder = OneHotEncoder(inputCol=f"{col}_indexed", outputCol=f"{col}_encoded")
            stages += [indexer, encoder]
            
        pipeline = Pipeline(stages=stages)
        model = pipeline.fit(df)
        return model.transform(df)

    def scale_features(self, df: DataFrame, input_col: str, output_col: str = "features_scaled", with_mean=True, with_std=True):

        scaler = StandardScaler(inputCol=input_col, outputCol=output_col, withMean=with_mean, withStd=with_std)
        model = scaler.fit(df)
        return model.transform(df)

    def assemble_features(self, df: DataFrame, input_cols: list, output_col: str = "features"):
        assembler = VectorAssembler(inputCols=input_cols, outputCol=output_col, handleInvalid="skip")
        return assembler.transform(df)

if __name__ == "__main__":
    from src.spark_utils import get_spark_session
    from src.data_loader import DataLoader
    
    spark = get_spark_session(app_name="FeatureEngineeringTest")
    loader = DataLoader(spark)
    
    df = loader.generate_synthetic_data(num_rows=100, num_features=3)
    
    engineer = FeatureEngineer(spark)
    
    feature_cols = ["feature_0", "feature_1", "feature_2"]
    df_assembled = engineer.assemble_features(df, input_cols=feature_cols)
    df_scaled = engineer.scale_features(df_assembled, input_col="features")
    
    df_scaled.select("features", "features_scaled").show(5, truncate=False)
