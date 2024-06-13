#!/bin/bash
set -e

# build custom docker spark image
sudo docker build -f Dockerfile.spark -t myspark .

chmod +777 datasets_temp metrics model

sudo docker-compose up --scale spark-worker=2

# submit script to master
sudo docker exec -it spark-master bin/spark-submit --master spark://spark-master:7077 /data/spark-scripts/load_dataset_and_train.py
