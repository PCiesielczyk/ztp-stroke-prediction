import logging
import os

import matplotlib.pyplot as plt
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.functions import vector_to_array
from pyspark.sql.functions import col, log

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(name)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S')
logger = logging.getLogger('metrics')

label = "label"


def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def print_and_plot_roc_curve(training_summary, predictions):
    evaluator = BinaryClassificationEvaluator(labelCol=label)
    area_under_curve = evaluator.evaluate(predictions)

    logger.info(f"Area under ROC curve: {area_under_curve}")

    lrROC = training_summary.roc.toPandas()

    create_directory('/data/metrics')

    plt.figure(figsize=(8, 6))
    plt.plot(lrROC['FPR'], lrROC['TPR'])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.title('ROC Curve')
    plt.savefig('/data/metrics/ROC_curve.png')


def print_accuracy_precision_recall_f1(training_summary):
    pr = training_summary.pr.toPandas()
    plt.figure(figsize=(8, 6))
    plt.plot(pr['recall'], pr['precision'])
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    plt.savefig('/data/metrics/precision_recall.png')

    accuracy = training_summary.accuracy
    falsePositiveRate = training_summary.weightedFalsePositiveRate
    truePositiveRate = training_summary.weightedTruePositiveRate
    fMeasure = training_summary.weightedFMeasure()
    precision = training_summary.weightedPrecision
    recall = training_summary.weightedRecall

    logger.info("Accuracy: %s\nFPR: %s\nTPR: %s\nF-measure: %s\nPrecision: %s\nRecall: %s"
                % (accuracy, falsePositiveRate, truePositiveRate, fMeasure, precision, recall))


def print_coefficients_intercept(model):
    logger.info("Coefficients: \n" + str(model.coefficientMatrix))
    logger.info("Intercept: " + str(model.interceptVector))


def print_loss(predictions):
    probability_positive = predictions.select(vector_to_array(col("probability")).alias("probability"))
    probability_positive = probability_positive.withColumn("probability", col("probability")[1])

    predictions = predictions.crossJoin(probability_positive.withColumnRenamed("probability", "probability_positive"))

    log_loss = predictions.withColumn(
        "log_loss",
        - (col("label") * log(col("probability_positive")) + (1 - col("label")) * log(1 - col("probability_positive")))
    )
    average_log_loss = log_loss.agg({"log_loss": "mean"}).collect()[0]["avg(log_loss)"]
    logger.info(f"Average loss: {average_log_loss}")


def print_metrics(model, predictions):
    training_summary = model.summary

    print_and_plot_roc_curve(training_summary, predictions)
    print_accuracy_precision_recall_f1(training_summary)
    print_coefficients_intercept(model)
    print_loss(predictions)
