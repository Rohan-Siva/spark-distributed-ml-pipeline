#!/bin/bash

spark-submit \
    --master local[*] \
    --name "DistributedMLPipeline" \
    --conf spark.executor.memory=2g \
    --conf spark.driver.memory=2g \
    src/pipeline.py "$@"
