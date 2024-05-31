import logging

from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit

from preprocessing import encode_dataframe
from metrics import print_metrics

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(name)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S')
logger = logging.getLogger('load_dataset_and_train')

if __name__ == '__main__':
    dataset_path = "dataset.csv"
    spark = SparkSession.builder.master("local").appName("WordCount").getOrCreate()
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

    logger.info("Training model and searching for best hyperparameters...")
    model = lr_model.fit(train_data)

    lr_predictions = model.transform(test_data)
    print_metrics(model, lr_predictions)

    spark.stop()
