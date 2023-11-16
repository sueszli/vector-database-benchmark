"""
 Use DataFrames and SQL to count words in UTF8 encoded, '\\n' delimited text received from the
 network every second.

 Usage: sql_network_wordcount.py <hostname> <port>
   <hostname> and <port> describe the TCP server that Spark Streaming would connect to receive data.

 To run this on your local machine, you need to first run a Netcat server
    `$ nc -lk 9999`
 and then run the example
    `$ bin/spark-submit examples/src/main/python/streaming/sql_network_wordcount.py localhost 9999`
"""
import sys
import datetime
from pyspark import SparkConf, SparkContext
from pyspark.rdd import RDD
from pyspark.streaming import StreamingContext
from pyspark.sql import Row, SparkSession

def getSparkSessionInstance(sparkConf: SparkConf) -> SparkSession:
    if False:
        while True:
            i = 10
    if 'sparkSessionSingletonInstance' not in globals():
        globals()['sparkSessionSingletonInstance'] = SparkSession.builder.config(conf=sparkConf).getOrCreate()
    return globals()['sparkSessionSingletonInstance']
if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: sql_network_wordcount.py <hostname> <port> ', file=sys.stderr)
        sys.exit(-1)
    (host, port) = sys.argv[1:]
    sc = SparkContext(appName='PythonSqlNetworkWordCount')
    ssc = StreamingContext(sc, 1)
    lines = ssc.socketTextStream(host, int(port))
    words = lines.flatMap(lambda line: line.split(' '))

    def process(time: datetime.datetime, rdd: RDD[str]) -> None:
        if False:
            return 10
        print('========= %s =========' % str(time))
        try:
            spark = getSparkSessionInstance(rdd.context.getConf())
            rowRdd = rdd.map(lambda w: Row(word=w))
            wordsDataFrame = spark.createDataFrame(rowRdd)
            wordsDataFrame.createOrReplaceTempView('words')
            wordCountsDataFrame = spark.sql('select word, count(*) as total from words group by word')
            wordCountsDataFrame.show()
        except BaseException:
            pass
    words.foreachRDD(process)
    ssc.start()
    ssc.awaitTermination()