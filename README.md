# Distributed ML Pipeline with Spark

## Overview
This project implements a robust, end-to-end distributed Machine Learning pipeline using Apache Spark and MLflow. It is designed to handle large-scale datasets, demonstrating scalable feature engineering, distributed model training, and production-ready MLOps workflows.

## Tools Used
- **Big Data Processing**: Apache Spark (PySpark)
- **Experiment Tracking**: MLflow
- **Data Manipulation**: Pandas, NumPy
- **Testing**: Pytest
- **Configuration**: PyYAML

## Structure
- `src/`: Contains the core source code for the pipeline, including modules for data loading, feature engineering, and model training.
- `config/`: Configuration files (YAML) for managing pipeline parameters.
- `notebooks/`: Jupyter notebooks for exploratory data analysis and prototyping.
- `tests/`: Unit tests to ensure code reliability.
- `deployment/`: Scripts and configurations for deploying the pipeline to production environments.

## Setup

### Prerequisites
- Java (JDK 8 or 11)
- Python 3.8+

### Installation
1. Clone the repository.
2. Install the required Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## How It Works
1. **Data Ingestion**: The pipeline loads data from distributed sources using Spark.
2. **Feature Engineering**: Scalable transformations are applied to the data to create features for modeling.
3. **Training**: Machine learning models are trained in a distributed manner using Spark MLlib.
4. **Tracking**: Experiments, parameters, and metrics are logged to MLflow for reproducibility.
5. **Deployment**: The trained models can be deployed using the scripts provided in the `deployment/` directory.

## Contact
For collaborations or questions, please reach out to rohansiva123@gmail.com.
