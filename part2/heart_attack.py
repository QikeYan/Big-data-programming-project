import sys
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.sql.types import *
from pyspark.ml.feature import (StandardScaler, 
                                OneHotEncoderEstimator, VectorAssembler)
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.sql import Row


def count_nulls(df):
  null_counts = []
  for col in df.dtypes:
    cname = col[0]
    ctype = col[1]
    if ctype != 'string':
      nulls = df.where(df[cname].isNull()).count()
      result = tuple([cname, nulls])
      if nulls != 0:
        null_counts.append(result)
      
  return null_counts

if __name__ == "__main__":
	spark = SparkSession\
		.builder\
		.appName('Heart Attack!')\
		.getOrCreate()

	sc = spark.sparkContext

	data_file = sys.argv[1] + "/framingham.csv"

	schema_fields = [
	  StructField("male", IntegerType(), True),
	  StructField("age", DoubleType(), True),
	  StructField("education", IntegerType(), True),
	  StructField("currentSmoker", IntegerType(), True),
	  StructField("cigsPerDay", IntegerType(), True),
	  StructField("BPMeds", IntegerType(), True),
	  StructField("prevalentStroke", IntegerType(), True),
	  StructField("prevalentHyp", IntegerType(), True),
	  StructField("diabetes", IntegerType(), True),
	  StructField("totChol", DoubleType(), True),
	  StructField("sysBP", DoubleType(), True),
	  StructField("diaBP", DoubleType(), True),
	  StructField("BMI", DoubleType(), True),
	  StructField("heartRate", DoubleType(), True),
	  StructField("glucose", DoubleType(), True),
	  StructField("TenYearCHD", IntegerType(), True)
	]

	data_schema = StructType(schema_fields)

	data = spark.read.load(data_file, format='csv',
		               schema=data_schema,
		               header=True,
		               nullValue='')

	cols_discard = ['BPMeds', 'prevalentStroke', 'diabetes']
	data = data.drop('BPMeds', 'prevalentStroke', 'diabetes')


	null_counts = count_nulls(data)
	null_counts

	data = data.dropna(how='all', subset=[cname for cname, _ in null_counts])

	while True:
	  if len(count_nulls(data)) != 0:
	    data = data.dropna(how='any', subset=[cname for cname, _ in null_counts])
	  else:
	    break

	# Train/Test split
	train, test = data.randomSplit([0.8, 0.2], seed=5)
	label = 'TenYearCHD'

	numberical_cols = ['age', 'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose']
	categorical_cols = ['education', 'cigsPerDay']
	stages = []
        
	# Normalization
	numerical_assembler = VectorAssembler(inputCols=numberical_cols, outputCol='numFeatures')
	scaler = StandardScaler(inputCol='numFeatures', outputCol='norm_cols',
		               withStd=True, withMean=True)
	stages +=[numerical_assembler, scaler]

	# One-hot Encoding
	encoder = OneHotEncoderEstimator(inputCols=categorical_cols,
		                         outputCols=['genderClassVec', 'educationClassVec'])
	stages += [encoder]

	# Final Assembly
	inputCols = ['norm_cols'] + ['male', 'currentSmoker', 'prevalentHyp'] + ['genderClassVec', 'educationClassVec']
	final_assembler = VectorAssembler(inputCols=inputCols, outputCol='features')
	stages += [final_assembler]

	# Instantiate Model - Logistic Regression
	lr = LogisticRegression(featuresCol='features', labelCol=label, regParam=0.1)
	stages += [lr]

        # Assemble pipeline
	pipeline = Pipeline(stages = stages)
	model = pipeline.fit(train)
	res = model.transform(test)

	prob_pred = res.select('TenYearCHD', 'prediction')

	prob_pred.write.csv(sys.argv[2], header=True)
