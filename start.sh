#!/bin/sh
/usr/local/hadoop/sbin/start-dfs.sh
/usr/local/spark/sbin/start-all.sh
/usr/local/hadoop/bin/hdfs dfsadmin -safemode leave
