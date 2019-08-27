import sys
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.sql.types import *
from pyspark.ml.feature import (StandardScaler, StringIndexer,
                                OneHotEncoderEstimator, VectorAssembler)

from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.classification import (LogisticRegression,
                                       DecisionTreeClassifier, RandomForestClassifier)
from pyspark.ml import Pipeline
from pyspark.sql import Row

def count_nulls(df):
  null_counts = []
  for col in df.dtypes:
    cname = col[0]
    ctype = col[1]
    if ctype == 'string':
      nulls = df.select(cname).where(df[cname] == ' ?').count()
      result = tuple([cname, nulls])
      if nulls != 0:
        null_counts.append(result)
      
  return null_counts

if __name__ == "__main__":
  spark = SparkSession\
          .builder\
          .appName('Who is Rich?')\
          .getOrCreate()
  
  train_file = sys.argv[1] + "train.csv"
  test_file = sys.argv[1] + "test.csv"

  schema_fields = [
    StructField("age", DoubleType(), True),
    StructField("workclass", StringType(), True),
    StructField("fnlwgt", DoubleType(), True),
    StructField("education", StringType(), True),
    StructField("education_num", DoubleType(), True),
    StructField("marital_status", StringType(), True),
    StructField("occupation", StringType(), True),
    StructField("relationship", StringType(), True),
    StructField("race", StringType(), True),
    StructField("sex", StringType(), True),
    StructField("capital_gain", DoubleType(), True),
    StructField("capital_loss", DoubleType(), True),
    StructField("hours_per_week", DoubleType(), True),
    StructField("native_country", StringType(), True),
    StructField("salary", StringType(), True)
  ]
  
  data_schema = StructType(schema_fields)
  
  train = spark.read.load(train_file, format='csv',
                          schema=data_schema,
                          header=False,
                          nullValue=' ?')
  
  test = spark.read.load(test_file, format='csv',
                         schema=data_schema,
                         header=False,
                         nullValue=' ?')
  
  train = train.fillna('Unknown')
  test = test.fillna('Unknown')
  
  label = 'salary'
  numerical_cols = ['age', 'fnlwgt', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']
  categorical_cols = ["workclass", "education", "marital_status", "occupation", "relationship", "race", "sex", "native_country"]
  stages = []
  
  # One hot encode categorical cols
  for cname in categorical_cols:
    string_idxer = StringIndexer(inputCol=cname, outputCol=cname+'Index' )
    encoder = OneHotEncoderEstimator(inputCols=[string_idxer.getOutputCol()],
                                     outputCols=[cname+'classVec'])
    stages += [string_idxer, encoder]
    
  # Convert labels (Slary) to 0 and 1
  label_idxer = StringIndexer(inputCol="salary", outputCol="label")
  stages += [label_idxer]
    
  # Standardize numberical cols
  numerical_assembler = VectorAssembler(inputCols=numerical_cols, outputCol='numFeatures')
  scaler = StandardScaler(inputCol='numFeatures', outputCol='norm_cols',
                          withStd=True, withMean=True)
  
  stages += [numerical_assembler, scaler]

  #----------------- Logistic Regression -----------------------
  # In the test data, there is one value less than training data
  stages_lr = stages.copy()

  inputCols = ['norm_cols'] + [cname+"classVec"  for cname in categorical_cols if cname != 'native_country'] 
  final_assembler = VectorAssembler(inputCols=inputCols, outputCol='features')
  stages_lr += [final_assembler]

  pipeline = Pipeline(stages=stages_lr)
  train_lr = pipeline.fit(train).transform(train)
  test_lr = pipeline.fit(test).transform(test)
  lr = LogisticRegression(featuresCol='features', labelCol='label').fit(train_lr)
  res_lr = lr.transform(test_lr)

  #----------------- Decision and Random Forest -----------------

  # Final assembly                                                                                                
  inputCols = ['norm_cols'] + [cname+"classVec" for cname in categorical_cols]
  final_assembler = VectorAssembler(inputCols=inputCols, outputCol='features')
  stages += [final_assembler]
  
  pipeline = Pipeline(stages=stages)
  train_final = pipeline.fit(train).transform(train)
  test_final = pipeline.fit(test).transform(test)
  
  dt = DecisionTreeClassifier(featuresCol='features', labelCol='label').fit(train_final)
  res_dt = dt.transform(test_final)
  
  rf = RandomForestClassifier(featuresCol='features', labelCol='label', numTrees=20).fit(train_final)
  res_rf = rf.transform(test_final)
  
  res_lr.select('prediction', 'label').write.csv(sys.argv[2]+"lr", header=True)
  res_dt.select('prediction', 'label').write.csv(sys.argv[2]+"dt", header=True)
  res_rf.select('prediction', 'label').write.csv(sys.argv[2]+"rf", header=True)
  
  spark.stop()
