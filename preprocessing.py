import logging

from pyspark.ml.feature import StringIndexer
from pyspark.sql import DataFrame
from pyspark.sql.functions import when, col
from pyspark.sql.types import DoubleType

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(name)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S')
logger = logging.getLogger('preprocessing')

string_columns = ["work_type", "smoking_status"]


def convert_to_numerical(df: DataFrame) -> DataFrame:
    indexers = [StringIndexer(inputCol=col_name, outputCol=col_name + "_index") for col_name in string_columns]
    indexerModels = [indexer.fit(df) for indexer in indexers]
    for indexerModel in indexerModels:
        df = indexerModel.transform(df)
    for string_column in string_columns:
        df = df.drop(string_column)
    return df


def oversample_minority_class(df: DataFrame) -> DataFrame:
    ratio = 1
    minority_count = df.filter(col('stroke') == 1).count()
    majority_count = df.filter(col('stroke') == 0).count()

    balance_ratio = majority_count / minority_count

    logger.info(f"Initial Majority:Minority ratio is {balance_ratio:.2f}:1")

    oversampled_minority = df.filter(col('stroke') == 1).sample(withReplacement=True, fraction=(balance_ratio / ratio),
                                                                seed=88)
    oversampled_df = df.filter(col('stroke') == 0).union(oversampled_minority)
    return oversampled_df


def encode_dataframe(df: DataFrame) -> DataFrame:
    df = df.withColumn("gender", when(df["gender"] == "Male", 1).otherwise(0))
    df = df.withColumn("ever_married", when(df["ever_married"] == "Yes", 1).otherwise(0))
    df = df.withColumn("Residence_type", when(df["Residence_type"] == "Urban", 1).otherwise(0))
    df = df.filter(df["bmi"] != "N/A")
    df = df.drop("id")

    df = convert_to_numerical(df)

    columns_double = [col(col_name).cast(DoubleType()).alias(col_name) for col_name in df.columns]
    df = df.select(columns_double)
    df = df.withColumnRenamed("stroke", "label")

    logger.info(f"Encoded dataset. Row count: {df.count()}")

    df = oversample_minority_class(df)
    logger.info(f"Oversampled minority class. Row count: {df.count()}")

    return df
