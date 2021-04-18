# Databricks notebook source
# MAGIC %md
# MAGIC ## Using Cross Validation
# MAGIC 
# MAGIC In this exercise, you will use cross-validation to optimize parameters for a regression model. cross-validation is an approach
# MAGIC Where instead of just splitting the data into two sets (training and test data), we pick a number which we call K and we make a
# MAGIC K number of folds in the data.
# MAGIC 
# MAGIC ### Why cross-validation: 
# MAGIC Where because you're only using one training set and one validation set, You could still end up over fitting your model that might not always produce the optimal model with the optimal parameters
# MAGIC 
# MAGIC ### Prepare the Data
# MAGIC 
# MAGIC First, import the libraries you will need and prepare the training and test data:

# COMMAND ----------

# Import Spark SQL and Spark ML libraries
from pyspark.sql.types import *
from pyspark.sql.functions import *

from pyspark.ml import Pipeline
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler, MinMaxScaler
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import RegressionEvaluator

from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession

IS_DB = True

# COMMAND ----------

# MAGIC %md
# MAGIC ### TODO 0: Run the code in PySpark CLI
# MAGIC 1. Set the following to True:
# MAGIC ```
# MAGIC PYSPARK_CLI = True
# MAGIC ```
# MAGIC 1. You need to generate py (Python) file: File > Export > Source File
# MAGIC 1. Run it at your Hadoop/Spark cluster:
# MAGIC ```
# MAGIC $ spark-submit Python_Regression_Cross_Validation.py
# MAGIC ```

# COMMAND ----------

PYSPARK_CLI = True
if PYSPARK_CLI:
    sc = SparkContext.getOrCreate()
    spark = SparkSession(sc)

# COMMAND ----------

# DataFrame Schema, that should be a Table schema by Jongwook Woo (jwoo5@calstatela.edu) 01/07/2016
flightSchema = StructType([
  StructField("DayofMonth", IntegerType(), False),
  StructField("DayOfWeek", IntegerType(), False),
  StructField("Carrier", StringType(), False),
  StructField("OriginAirportID", IntegerType(), False),
  StructField("DestAirportID", IntegerType(), False),
  StructField("DepDelay", IntegerType(), False),
  StructField("ArrDelay", IntegerType(), False),
])

# COMMAND ----------

# MAGIC %md
# MAGIC ## TODO 1. Locate the flights.csv file

# COMMAND ----------

# File location and type
file_location = "/user/aelahi/cis5560/flights.csv" 
file_type = "csv"

# CSV options
infer_schema = "false"
first_row_is_header = "true"
delimiter = ","

# The applied options are for CSV files. For other file types, these will be ignored.
df = spark.read.format(file_type) \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load(file_location)

#display(df)
 df.show(10)
# COMMAND ----------

# MAGIC %md
# MAGIC ## TODO 2: Create a temporary view of the dataframe df

# COMMAND ----------

# Create a view or table

temp_table_name = "flights_csv"

df.createOrReplaceTempView(temp_table_name)

csv = df 
csv.show(5) 

# COMMAND ----------


# COMMAND ----------

# Select features and label
data = csv.select("DayofMonth", "DayOfWeek", "OriginAirportID", "DestAirportID", "DepDelay", col("ArrDelay").alias("label"))

# Split the data
splits = data.randomSplit([0.7, 0.3])
train = splits[0]
test = splits[1].withColumnRenamed("label", "trueLabel")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Define the Pipeline
# MAGIC Now define a pipeline that creates a feature vector and trains a regression model

# COMMAND ----------

# Define the pipeline
assembler = VectorAssembler(inputCols = ["DayofMonth", "DayOfWeek", "OriginAirportID", "DestAirportID", "DepDelay"], outputCol="features")
# minMax Scale; number vector is normalized: 04/20/2021
minMax = MinMaxScaler(inputCol = assembler.getOutputCol(), outputCol="normFeatures")

#lr = LogisticRegression(labelCol="label", featuresCol="features")
lr = LinearRegression(labelCol="label", featuresCol="normFeatures")
#pipeline = Pipeline(stages=[assembler, lr])
pipeline = Pipeline(stages=[assembler, minMax, lr])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Tune Parameters
# MAGIC You can tune parameters to find the best model for your data. To do this you can use the  **CrossValidator** class to evaluate each combination of parameters defined in a **ParameterGrid** against multiple *folds* of the data split into training and validation datasets, in order to find the best performing parameters. Note that this can take a long time to run because every parameter combination is tried multiple times.

# COMMAND ----------

paramGrid = ParamGridBuilder().addGrid(lr.regParam, [0.3, 0.01]).addGrid(lr.maxIter, [10, 5]).build()
# TODO: K = 2, you may test it with 5, 10
# K=2, 3, 5, 10: Root Mean Square Error (RMSE): 13.2
K = 2
cv = CrossValidator(estimator=pipeline, evaluator=RegressionEvaluator(), estimatorParamMaps=paramGrid, numFolds=K)

model = cv.fit(train)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Test the Model
# MAGIC Now you're ready to apply the model to the test data.

# COMMAND ----------

prediction = model.transform(test)
#predicted = prediction.select("features", "prediction", "trueLabel")
predicted = prediction.select("normFeatures", "prediction", "trueLabel")
predicted.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Examine the Predicted and Actual Values
# MAGIC You can plot the predicted values against the actual values to see how accurately the model has predicted. In a perfect model, the resulting scatter plot should form a perfect diagonal line with each predicted value being identical to the actual value - in practice, some variance is to be expected.
# MAGIC Run the cells below to create a temporary table from the **predicted** DataFrame and then retrieve the predicted and actual label values using SQL. You can then display the results as a scatter plot, specifying **-** as the function to show the unaggregated values.

# COMMAND ----------

predicted.createOrReplaceTempView("regressionPredictions")

# COMMAND ----------

# Microsoft Azure for data visualization
'''
%%sql
SELECT trueLabel, prediction FROM regressionPredictions
'''

# COMMAND ----------

# Reference: http://standarderror.github.io/notes/Plotting-with-PySpark/
dataPred = spark.sql("SELECT trueLabel, prediction FROM regressionPredictions")
## Need it for Databricks
display(dataPred)

# COMMAND ----------

# MAGIC %md
# MAGIC ### The following is for IBM Watson not Databricks

# COMMAND ----------

# IBM Data Science with matplotlib for data visualization

## Need the following for IBM Watson Studio
#%matplotlib inline
#import pandas as pd
#import matplotlib
#import matplotlib.pyplot as plt
import numpy as np

if IS_DB: 
  ## Need the following for IBM Watson Studio
  #from pandas.tools.plotting import scatter_matrix

  # Reference: http://standarderror.github.io/notes/Plotting-with-PySpark/
  dataPred = spark.sql("SELECT trueLabel, prediction FROM regressionPredictions")
  # convert to pandas and plot
  ## Need the following for IBM Watson Studio
  # regressionPredictionsPanda = dataPred.toPandas()
  # stuff = scatter_matrix(regressionPredictionsPanda, alpha=0.7, figsize=(6, 6), diagonal='kde')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Retrieve the Root Mean Square Error (RMSE)
# MAGIC There are a number of metrics used to measure the variance between predicted and actual values. Of these, the root mean square error (RMSE) is a commonly used value that is measured in the same units as the prediced and actual values - so in this case, the RMSE indicates the average number of minutes between predicted and actual flight delay values. You can use the **RegressionEvaluator** class to retrieve the RMSE (about 13.24 with 10 folds).

# COMMAND ----------

evaluator = RegressionEvaluator(labelCol="trueLabel", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(prediction)
print ("Root Mean Square Error (RMSE):", rmse)

# COMMAND ----------

# MAGIC %md
# MAGIC ### References
# MAGIC 1. Class Imbalance in Credit Card Fraud Detection - Part 3 : Undersampling in Spark, http://blog.madhukaraphatak.com/class-imbalance-part-3/
# MAGIC 1. Winning a Kaggle competition with Apache Spark and SparkML Machine Learning Pipelines, https://developer.ibm.com/tv/dwlive010-replay-code-machine-learning-flow-spark-ml/
# MAGIC 1. Amazon S3 with Apache Spark, https://docs.databricks.com/spark/latest/data-sources/aws/amazon-s3.html
# MAGIC 1. How to create and query a table or DataFrame on AWS S3, https://docs.databricks.com/_static/notebooks/data-import/s3.html
# MAGIC 1. https://github.com/romeokienzler/uhack/tree/master/projects/bosch
# MAGIC 1. Access DBFS with dbutils, https://docs.databricks.com/user-guide/dbfs-databricks-file-system.html#access-dbfs-with-dbutils

# COMMAND ----------


