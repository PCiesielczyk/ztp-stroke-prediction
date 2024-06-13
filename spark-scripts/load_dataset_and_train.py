import logging
import time

from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType

from metrics import print_metrics
from preprocessing import encode_dataframe

POLL_INTERVAL = "60 seconds"

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(name)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S')
logger = logging.getLogger('load_dataset_and_train')

dataset_schema = StructType([
    StructField("id", StringType(), True),
    StructField("gender", StringType(), True),
    StructField("age", StringType(), True),
    StructField("hypertension", StringType(), True),
    StructField("heart_disease", StringType(), True),
    StructField("ever_married", StringType(), True),
    StructField("work_type", StringType(), True),
    StructField("Residence_type", StringType(), True),
    StructField("avg_glucose_level", StringType(), True),
    StructField("bmi", StringType(), True),
    StructField("smoking_status", StringType(), True),
    StructField("stroke", StringType(), True)
])

spark = SparkSession.builder.appName("WordCount").getOrCreate()


def process_stream(data_frame, epoch_id):
    dataset_path = "/data/datasets_temp/batch"
    data_frame.write \
        .format("csv") \
        .mode("overwrite") \
        .save(dataset_path)

    df = spark.read.option("header", True).csv(dataset_path)

    logger.info(f"Loaded {dataset_path}. Row count: {df.count()}, {len(df.columns)} columns in the dataset.")

    label = "label"
    df = encode_dataframe(df)
    features = [col for col in df.columns if col not in [label]]
    logger.info(f"Features columns: {features}")

    stages_list = []

    assembler = VectorAssembler(inputCols=features, outputCol="features")

    stages_list += [assembler]
    pipeline = Pipeline(stages=stages_list)
    df_pipeline = pipeline.fit(df)
    df_transformed = df_pipeline.transform(df)

    train_data, test_data = df_transformed.randomSplit([0.8, 0.2], seed=123)

    lr_model = LogisticRegression(
        featuresCol="features",
        family="binomial",
        labelCol=label
    )

    logger.info("Training model...")
    model = lr_model.fit(train_data)

    lr_predictions = model.transform(test_data)
    print_metrics(model, lr_predictions)


if __name__ == '__main__':
    streaming_df = spark.readStream \
        .schema(dataset_schema) \
        .csv("/data/datasets")

    query = streaming_df.writeStream \
        .foreachBatch(process_stream) \
        .trigger(processingTime=POLL_INTERVAL) \
        .start()
    query.awaitTermination()
    spark.stop()
