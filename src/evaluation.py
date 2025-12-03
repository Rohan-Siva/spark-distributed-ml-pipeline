from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator, RegressionEvaluator
from pyspark.sql import DataFrame
import mlflow

class ModelEvaluator:
    def __init__(self, spark):
        self.spark = spark

    def evaluate(self, predictions: DataFrame, label_col="label", metric_name="areaUnderROC", model_type="classification"):

        if model_type == "classification":
            if metric_name in ["areaUnderROC", "areaUnderPR"]:
                evaluator = BinaryClassificationEvaluator(labelCol=label_col, metricName=metric_name)
            else:
                evaluator = MulticlassClassificationEvaluator(labelCol=label_col, metricName=metric_name)
        elif model_type == "regression":
            evaluator = RegressionEvaluator(labelCol=label_col, metricName=metric_name)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
            
        metric_value = evaluator.evaluate(predictions)
        
        # Log to MLflow if active run exists
        if mlflow.active_run():
            mlflow.log_metric(metric_name, metric_value)
            
        return metric_value
