#!/bin/bash

source ../env.sh

../start.sh

/usr/local/hadoop/bin/hdfs dfs -rm -r /part1/input/
/usr/local/hadoop/bin/hdfs dfs -rm -r /part1/output/
/usr/local/hadoop/bin/hdfs dfs -mkdir -p /part1/input/
/usr/local/hadoop/bin/hdfs dfs -copyFromLocal ./data/train.csv /part1/input/
/usr/local/hadoop/bin/hdfs dfs -copyFromLocal ./data/test.csv /part1/input/
/usr/local/spark/bin/spark-submit --master=spark://$SPARK_MASTER:7077 toxic_comment.py hdfs://$SPARK_MASTER:9000/part1/input/ hdfs://$SPARK_MASTER:9000/part1/output/

/usr/local/hadoop/bin/hdfs dfs -get /part1/output/ /project2/part1/results
/usr/local/hadoop/bin/hdfs dfs -rm -r /part1/input/
/usr/local/hadoop/bin/hdfs dfs -rm -r /part1/output/

../stop.sh


