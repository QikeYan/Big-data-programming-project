#!/bin/bash

source ../env.sh

../start.sh

/usr/local/hadoop/bin/hdfs dfs -rm -r /part3/input/
/usr/local/hadoop/bin/hdfs dfs -rm -r /part3/output/
/usr/local/hadoop/bin/hdfs dfs -mkdir -p /part3/input/
/usr/local/hadoop/bin/hdfs dfs -copyFromLocal ./data/train.csv /part3/input/
/usr/local/hadoop/bin/hdfs dfs -copyFromLocal ./data/test.csv /part3/input/

/usr/local/spark/bin/spark-submit --master=spark://$SPARK_MASTER:7077 who_rich.py hdfs://$SPARK_MASTER:9000/part3/input/ hdfs://$SPARK_MASTER:9000/part3/output/

/usr/local/hadoop/bin/hdfs dfs -get /part3/output/ /project2/part3/results
/usr/local/hadoop/bin/hdfs dfs -rm -r /part3/input/
/usr/local/hadoop/bin/hdfs dfs -rm -r /part3/output/

../stop.sh


