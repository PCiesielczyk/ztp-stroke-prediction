docker build -f Dockerfile.spark -t myspark .
docker-compose up -d --scale spark-worker=2
docker exec -it spark-master bin/spark-submit --master spark://spark-master:7077 /data/spark-scripts/load_dataset_and_train.py
