from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier
from pyspark.ml.regression import LinearRegression, RandomForestRegressor, GBTRegressor
from pyspark.ml.evaluation import BinaryClassificationEvaluator, RegressionEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql import DataFrame

class ModelTrainer:
    def __init__(self, spark):
        self.spark = spark

    def train_model(self, df: DataFrame, model_type="logistic_regression", label_col="label", features_col="features", experiment_name=None):
        import mlflow
        from mlflow.spark import log_model
        
        if experiment_name:
            mlflow.set_experiment(experiment_name)
            
        with mlflow.start_run(run_name=f"train_{model_type}"):
            mlflow.log_param("model_type", model_type)
            mlflow.log_param("label_col", label_col)
            
            if model_type == "logistic_regression":
                lr = LogisticRegression(labelCol=label_col, featuresCol=features_col)
                model = lr.fit(df)
                mlflow.log_param("regParam", lr.getRegParam())
                mlflow.log_param("elasticNetParam", lr.getElasticNetParam())
            elif model_type == "random_forest":
                rf = RandomForestClassifier(labelCol=label_col, featuresCol=features_col)
                model = rf.fit(df)
                mlflow.log_param("numTrees", rf.getNumTrees())
                mlflow.log_param("maxDepth", rf.getMaxDepth())
            elif model_type == "gbt":
                gbt = GBTClassifier(labelCol=label_col, featuresCol=features_col)
                model = gbt.fit(df)
                mlflow.log_param("maxIter", gbt.getMaxIter())
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            # Log model
            log_model(model, "model")
            return model

    def tune_hyperparameters(self, df: DataFrame, model_type="logistic_regression", label_col="label", features_col="features", num_folds=3, experiment_name=None):
        import mlflow
        
        if experiment_name:
            mlflow.set_experiment(experiment_name)
            
        with mlflow.start_run(run_name=f"tune_{model_type}"):
            mlflow.log_param("model_type", model_type)
            mlflow.log_param("num_folds", num_folds)
            
            if model_type == "logistic_regression":
                estimator = LogisticRegression(labelCol=label_col, featuresCol=features_col)
                param_grid = ParamGridBuilder() \
                    .addGrid(estimator.regParam, [0.1, 0.01]) \
                    .addGrid(estimator.elasticNetParam, [0.0, 0.5, 1.0]) \
                    .build()
                evaluator = BinaryClassificationEvaluator(labelCol=label_col)
            elif model_type == "random_forest":
                estimator = RandomForestClassifier(labelCol=label_col, featuresCol=features_col)
                param_grid = ParamGridBuilder() \
                    .addGrid(estimator.numTrees, [10, 20]) \
                    .addGrid(estimator.maxDepth, [5, 10]) \
                    .build()
                evaluator = BinaryClassificationEvaluator(labelCol=label_col)
            else:
                raise ValueError(f"Unsupported model type for tuning: {model_type}")
                
            cv = CrossValidator(estimator=estimator,
                                estimatorParamMaps=param_grid,
                                evaluator=evaluator,
                                numFolds=num_folds)
                                
            cv_model = cv.fit(df)
            best_model = cv_model.bestModel
            
           
            mlflow.spark.log_model(best_model, "best_model")
            
            return best_model

    def evaluate_model(self, model, df: DataFrame, label_col="label", metric_name="areaUnderROC"):
        
        predictions = model.transform(df)
        
        if metric_name in ["areaUnderROC", "areaUnderPR"]:
            evaluator = BinaryClassificationEvaluator(labelCol=label_col, metricName=metric_name)
        elif metric_name in ["accuracy", "f1", "weightedPrecision", "weightedRecall"]:
            evaluator = MulticlassClassificationEvaluator(labelCol=label_col, metricName=metric_name)
        elif metric_name in ["rmse", "mse", "r2", "mae"]:
            evaluator = RegressionEvaluator(labelCol=label_col, metricName=metric_name)
        else:
            raise ValueError(f"Unsupported metric: {metric_name}")
            
        return evaluator.evaluate(predictions)

if __name__ == "__main__":
    from src.spark_utils import get_spark_session
    from src.data_loader import DataLoader
    from src.feature_engineering import FeatureEngineer
    
    spark = get_spark_session(app_name="ModelTrainingTest")
    loader = DataLoader(spark)
    engineer = FeatureEngineer(spark)
    
    
    df = loader.generate_synthetic_data(num_rows=200, num_features=5)
    feature_cols = [f"feature_{i}" for i in range(5)]
    df_assembled = engineer.assemble_features(df, input_cols=feature_cols)
    
    
    train_df, test_df = df_assembled.randomSplit([0.8, 0.2], seed=42)
    
    trainer = ModelTrainer(spark)
    
    
    model = trainer.train_model(train_df, model_type="logistic_regression")
    auc = trainer.evaluate_model(model, test_df)
    print(f"Logistic Regression AUC: {auc}")
    
    
    print("Tuning Logistic Regression...")
    best_model = trainer.tune_hyperparameters(train_df, model_type="logistic_regression", num_folds=2)
    auc_tuned = trainer.evaluate_model(best_model, test_df)
    print(f"Tuned Logistic Regression AUC: {auc_tuned}")
