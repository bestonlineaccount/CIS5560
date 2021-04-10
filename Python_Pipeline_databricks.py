# Databricks notebook source
# MAGIC %md ## CIS5560: PySpark Pipeline Classification in in Databricks
# MAGIC 
# MAGIC ### Jongwook Woo (jwoo5@calstatela.edu), revised on 03/22/2019, 04/08/2018
# MAGIC Tested in Databricks CE's default cluster (5.2, includes Apache Spark 2.4.0, Scala 2.11, Python 3)

# COMMAND ----------

# MAGIC %md ## Creating a Pipeline
# MAGIC 
# MAGIC In this exercise, you will implement a pipeline that includes multiple stages of *transformers* and *estimators* to prepare features and train a classification model. The resulting trained *PipelineModel* can then be used as a transformer to predict whether or not a flight will be late.
# MAGIC 
# MAGIC ### Import Spark SQL and Spark ML Libraries
# MAGIC 
# MAGIC First, import the libraries you will need:

# COMMAND ----------

from pyspark.sql.types import *
from pyspark.sql.functions import *

from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import VectorAssembler, StringIndexer, VectorIndexer, MinMaxScaler

# COMMAND ----------

# MAGIC %md ### Load Source Data
# MAGIC The data for this exercise is provided as a CSV file containing details of flights. The data includes specific characteristics (or *features*) for each flight, as well as a column indicating how many minutes late or early the flight arrived.
# MAGIC 
# MAGIC You will load this data into a DataFrame and display it.

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

# MAGIC %md ### Read csv file from DBFS (Databricks File Systems)
# MAGIC 
# MAGIC ### TODO 1: follow the direction to read your table after upload it to Data at the left frame
# MAGIC ### NOTE: See reference [[1](https://docs.databricks.com/user-guide/tables.html#create-a-table)]
# MAGIC 1. After flights.csv file is added to the data of the left frame, create a table using the UI, especially, "Upload File"
# MAGIC 1. Click "Preview Table to view the table" and Select the option as flights.csv has a header as the first row: "First line is header"
# MAGIC 1. Change the data type of the table columns as shown in flightSchema of the above cell
# MAGIC 1. When you click on create table button, remember the table name, for example, _flights_csv_

# COMMAND ----------

# MAGIC %fs ls /FileStore/tables/flights.csv

# COMMAND ----------

# MAGIC %md ### TODO 2: Assign the table name to data, which is created at TODO 1, using Spark SQL 
# MAGIC #### _spark.sql("SELECT * FROM flights_csv")_, 
# MAGIC csv = spark.sql("SELECT * FROM flights_csv")

# COMMAND ----------

#csv = spark.sql("SELECT * FROM flights_csv")
#csv = spark.sql("SELECT * FROM flights_spr2019")
csv = spark.read.csv('/user/jwoo5/flights.csv', inferSchema=True, header=True)

# Microsoft Azure 
#csv = spark.read.csv('wasb:///data/flights.csv', inferSchema=True, header=True)
#csv.show()

csv.show(5)

# COMMAND ----------

# MAGIC %md ### Prepare the Data
# MAGIC Most modeling begins with exhaustive exploration and preparation of the data. In this example, the data has been cleaned for you. You will simply select a subset of columns to use as *features* and create a Boolean *label* field named **label** with the value **1** for flights that arrived 15 minutes or more after the scheduled arrival time, or **0** if the flight was early or on-time.

# COMMAND ----------

data = csv.select("DayofMonth", "DayOfWeek", "Carrier", "OriginAirportID", "DestAirportID", "DepDelay", ((col("ArrDelay") > 15).cast("Double").alias("label")))
# data = csv
data.show()

# COMMAND ----------

# MAGIC %md ### Split the Data
# MAGIC It is common practice when building supervised machine learning models to split the source data, using some of it to train the model and reserving some to test the trained model. In this exercise, you will use 70% of the data for training, and reserve 30% for testing. In the testing data, the **label** column is renamed to **trueLabel** so you can use it later to compare predicted labels with known actual values.

# COMMAND ----------

splits = data.randomSplit([0.7, 0.3])
train = splits[0]
test = splits[1].withColumnRenamed("label", "trueLabel")
train_rows = train.count()
test_rows = test.count()
print("Training Rows:", train_rows, " Testing Rows:", test_rows)

# COMMAND ----------

# MAGIC %md ### Define the Pipeline
# MAGIC A predictive model often requires multiple stages of feature preparation. For example, it is common when using some algorithms to distingish between continuous features (which have a calculable numeric value) and categorical features (which are numeric representations of discrete categories). It is also common to *normalize* continuous numeric features to use a common scale (for example, by scaling all numbers to a proportinal decimal value between 0 and 1).
# MAGIC 
# MAGIC A pipeline consists of a a series of *transformer* and *estimator* stages that typically prepare a DataFrame for
# MAGIC modeling and then train a predictive model. In this case, you will create a pipeline with seven stages:
# MAGIC - A **StringIndexer** estimator that converts string values to indexes for categorical features
# MAGIC - A **VectorAssembler** that combines categorical features into a single vector
# MAGIC - A **VectorIndexer** that creates indexes for a vector of categorical features
# MAGIC - A **VectorAssembler** that creates a vector of continuous numeric features
# MAGIC - A **MinMaxScaler** that normalizes continuous numeric features
# MAGIC - A **VectorAssembler** that creates a vector of categorical and continuous features
# MAGIC - A **DecisionTreeClassifier** that trains a classification model.

# COMMAND ----------

strIdx = StringIndexer(inputCol = "Carrier", outputCol = "CarrierIdx")

# the following columns are categorical number such as ID so that it should be Category features
catVect = VectorAssembler(inputCols = ["CarrierIdx", "DayofMonth", "DayOfWeek", "OriginAirportID", "DestAirportID"], outputCol="catFeatures")
catIdx = VectorIndexer(inputCol = catVect.getOutputCol(), outputCol = "idxCatFeatures")

# COMMAND ----------

# number is meaningful so that it should be number features
numVect = VectorAssembler(inputCols = ["DepDelay"], outputCol="numFeatures")
# number vector is normalized
minMax = MinMaxScaler(inputCol = numVect.getOutputCol(), outputCol="normFeatures")

featVect = VectorAssembler(inputCols=["idxCatFeatures", "normFeatures"], outputCol="features")
dt = DecisionTreeClassifier(labelCol="label", featuresCol="features")

# Pipeline process the series of transformation above, which is 7 transformation
pipeline = Pipeline(stages=[strIdx, catVect, catIdx, numVect, minMax, featVect, dt])

# COMMAND ----------

# MAGIC %md ### Run the Pipeline as an Estimator
# MAGIC The pipeline itself is an estimator, and so it has a **fit** method that you can call to run the pipeline on a specified DataFrame. In this case, you will run the pipeline on the training data to train a model.

# COMMAND ----------

# pipeline we implement can train a model
piplineModel = pipeline.fit(train)
print ("Pipeline complete!")

# COMMAND ----------

# MAGIC %md ### Test the Pipeline Model
# MAGIC The model produced by the pipeline is a transformer that will apply all of the stages in the pipeline to a specified DataFrame and apply the trained model to generate predictions. In this case, you will transform the **test** DataFrame using the pipeline to generate label predictions.

# COMMAND ----------

# piplineModel with train data set applies test data set and generate predictions
prediction = piplineModel.transform(test)
predicted = prediction.select("features", "prediction", "trueLabel")
predicted.show(100, truncate=False)

# COMMAND ----------

# MAGIC %md The resulting DataFrame is produced by applying all of the transformations in the pipline to the test data. The **prediction** column contains the predicted value for the label, and the **trueLabel** column contains the actual known value from the testing data.
