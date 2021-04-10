# Databricks notebook source
# MAGIC %md ## CIS5560: PySpark Pipeline Text Sentiment Analysis in Databricks
# MAGIC 
# MAGIC ### Jongwook Woo (jwoo5@calstatela.edu), revised on 04/08/2018
# MAGIC Tested in Databricks CE's default cluster (3.5LTS, Spark 2.2.1 Scala 2.11)

# COMMAND ----------

# MAGIC %md ## Text Analysis
# MAGIC In this lab, you will create a classification model that performs sentiment analysis of tweets.
# MAGIC ### Import Spark SQL and Spark ML Libraries
# MAGIC 
# MAGIC First, import the libraries you will need:

# COMMAND ----------

from pyspark.sql.types import *
from pyspark.sql.functions import *

from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import HashingTF, Tokenizer, StopWordsRemover

# COMMAND ----------

# MAGIC %md ### Load Source Data
# MAGIC Now load the tweets data into a DataFrame. This data consists of tweets that have been previously captured and classified as positive or negative.

# COMMAND ----------

# MAGIC %md ### Read csv file from DBFS (Databricks File Systems)
# MAGIC 
# MAGIC ### TODO 1: follow the direction to read your table after upload it to Data at the left frame
# MAGIC ### NOTE: See reference [[1](https://docs.databricks.com/user-guide/tables.html#create-a-table)]
# MAGIC 1. After flights.csv file is added to the data of the left frame, create a table using the UI, especially, "Upload File"
# MAGIC 1. Click "Preview Table to view the table" and Select the option as flights.csv has a header as the first row: "First line is header"
# MAGIC 1. Change the data type of the table columns as shown in flightSchema of the above cell
# MAGIC 1. When you click on create table button, remember the table name, for example, _tweets_csv_

# COMMAND ----------

# MAGIC %fs ls /FileStore/tables/tweets.csv

# COMMAND ----------

# MAGIC %md ### TODO 2: Assign the table name to data, which is created at TODO 1, using Spark SQL 
# MAGIC #### _spark.sql("SELECT * FROM tweets_csv")_, 
# MAGIC data = spark.sql("SELECT * FROM tweets_csv")

# COMMAND ----------

# Microsoft Azure 
# tweets_csv = spark.read.csv('wasb:///data/tweets.csv', inferSchema=True, header=True)
# tweets_csv.show(truncate = False)

# tweets_csv = spark.sql("SELECT * FROM tweets_csv")
tweets_csv = spark.read.csv('/user/jwoo5/tweets.csv', inferSchema=True, header=True)


# COMMAND ----------

# MAGIC %md ### Prepare the Data
# MAGIC The features for the classification model will be derived from the tweet text. The label is the sentiment (1 for positive, 0 for negative)

# COMMAND ----------

data = tweets_csv.select("SentimentText", col("Sentiment").cast("Int").alias("label"))
data.show(truncate = False)

# COMMAND ----------

# MAGIC %md ### Split the Data
# MAGIC In common with most classification modeling processes, you'll split the data into a set for training, and a set for testing the trained model.

# COMMAND ----------

splits = data.randomSplit([0.7, 0.3])
train = splits[0]
test = splits[1].withColumnRenamed("label", "trueLabel")
train_rows = train.count()
test_rows = test.count()
print("Training Rows:", train_rows, " Testing Rows:", test_rows)

# COMMAND ----------

# MAGIC %md ### Define the Pipeline
# MAGIC The pipeline for the model consist of the following stages:
# MAGIC - A Tokenizer to split the tweets into individual words.
# MAGIC - A StopWordsRemover to remove common words such as "a" or "the" that have little predictive value.
# MAGIC - A HashingTF class to generate numeric vectors from the text values.
# MAGIC - A LogisticRegression algorithm to train a binary classification model.

# COMMAND ----------

# convert sentence to words' list
tokenizer = Tokenizer(inputCol="SentimentText", outputCol="SentimentWords")
# remove stop words
swr = StopWordsRemover(inputCol=tokenizer.getOutputCol(), outputCol="MeaningfulWords")
# convert word to number as word frequency
hashTF = HashingTF(inputCol=swr.getOutputCol(), outputCol="features")
# set the model
lr = LogisticRegression(labelCol="label", featuresCol="features", maxIter=10, regParam=0.01)

# process pipeline with the series of transforms - 4 transforms
pipeline = Pipeline(stages=[tokenizer, swr, hashTF, lr])

# COMMAND ----------

# MAGIC %md ### Run the Pipeline as an Estimator
# MAGIC The pipeline itself is an estimator, and so it has a **fit** method that you can call to run the pipeline on a specified DataFrame. In this case, you will run the pipeline on the training data to train a model.

# COMMAND ----------

piplineModel = pipeline.fit(train)
print ("Pipeline complete!")

# COMMAND ----------

# MAGIC %md ### Test the Pipeline Model
# MAGIC The model produced by the pipeline is a transformer that will apply all of the stages in the pipeline to a specified DataFrame and apply the trained model to generate predictions. In this case, you will transform the **test** DataFrame using the pipeline to generate label predictions.

# COMMAND ----------

prediction = piplineModel.transform(test)
predicted = prediction.select("SentimentText", "prediction", "trueLabel")
predicted.show(100, truncate = False)

# COMMAND ----------

predicted10 = prediction.select("*")
predicted10.show(10, truncate = False)
