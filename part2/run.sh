#!/bin/bash

source ../env.sh

../start.sh

/usr/local/hadoop/bin/hdfs dfs -rm -r /part2/input/
/usr/local/hadoop/bin/hdfs dfs -rm -r /part2/output/
/usr/local/hadoop/bin/hdfs dfs -mkdir -p /part2/input/
/usr/local/hadoop/bin/hdfs dfs -copyFromLocal ./data/framingham.csv /part2/input/
/usr/local/spark/bin/spark-submit --master=spark://$SPARK_MASTER:7077 heart_attack.py hdfs://$SPARK_MASTER:9000/part2/input/ hdfs://$SPARK_MASTER:9000/part2/output/

/usr/local/hadoop/bin/hdfs dfs -get /part2/output/ /project2/part2/results
/usr/local/hadoop/bin/hdfs dfs -rm -r /part2/input/
/usr/local/hadoop/bin/hdfs dfs -rm -r /part2/output/

../stop.sh


