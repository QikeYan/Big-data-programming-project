import sys
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.ml.feature import Tokenizer, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.sql import Row

def parse_input(line):
  cols = line.split(',')
  res = []
  if len(cols) > 9:
    text = cols[2:-6]
    text = ','.join(text)
    res.append(cols[1])
    res.append(text)
    res += cols[-6:]
  else:
    res = cols[1:]
  return res

if __name__ == "__main__":
    spark = SparkSession\
            .builder\
            .appName('Toxic Comment Classification') \
            .getOrCreate()

    sc = spark.sparkContext

    train_file = sys.argv[1] + "/train.csv"
    test_file = sys.argv[1] + "/test.csv"

    train = sc.textFile(train_file, 10)
    test = sc.textFile(test_file, 10)
    
    train_cols = train.first()
    test_cols = test.first()
    
    out_cols = [i for i in train_cols.split(',') if i not in ["", "id", "comment_text"]]
    
    train_rdd = train.filter(lambda x: x != train_cols)\
                     .map(parse_input)\
                     .map(lambda p: Row(id=p[0],
                                        comment_text=p[1],
                                        toxic=int(p[2]),
                                        severe_toxic=int(p[3]),
                                        obscene=int(p[4]),
                                        threat=int(p[5]),
                                        insult=int(p[6]),
                                        identity_hate=int(p[7])
                     ))
    
    test_rdd = test.filter(lambda x: x != test_cols)\
                   .map(parse_input)\
                   .map(lambda p: Row(id=p[0],
                                      comment_text=p[1]))
    
    train_df = spark.createDataFrame(train_rdd)
    test_df = spark.createDataFrame(test_rdd)
    
    tok = Tokenizer(inputCol='comment_text', outputCol='words')
    htf = HashingTF(inputCol='words', outputCol='rawFeatures')
    idf = IDF(inputCol='rawFeatures', outputCol='features')
    
    extract_prob = F.udf(lambda x: float(x[1]), T.FloatType())
    
    final_res = test_df.select('id')
    for out_col in out_cols:
        lr = LogisticRegression(featuresCol='features', labelCol=out_col, regParam=0.1)
        
        pipeline = Pipeline(stages=[tok, htf, idf, lr])
        
        model = pipeline.fit(train_df)
        res = model.transform(test_df)
        
        final_res = final_res.join(res.select('id', 'probability'), on='id')
        final_res = final_res.withColumn(out_col, extract_prob('probability')).drop('probability')  
        
    final_res.write.csv(sys.argv[2], header=True) 
