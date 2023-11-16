"""
 Shows the most positive words in UTF8 encoded, '\\n' delimited text directly received the network
 every 5 seconds. The streaming data is joined with a static RDD of the AFINN word list
 (http://neuro.imm.dtu.dk/wiki/AFINN)

 Usage: network_wordjoinsentiments.py <hostname> <port>
   <hostname> and <port> describe the TCP server that Spark Streaming would connect to receive data.

 To run this on your local machine, you need to first run a Netcat server
    `$ nc -lk 9999`
 and then run the example
    `$ bin/spark-submit examples/src/main/python/streaming/network_wordjoinsentiments.py \\
    localhost 9999`
"""
import sys
from typing import Tuple
from pyspark import SparkContext
from pyspark.rdd import RDD
from pyspark.streaming import DStream, StreamingContext

def print_happiest_words(rdd: RDD[Tuple[float, str]]) -> None:
    if False:
        return 10
    top_list = rdd.take(5)
    print('Happiest topics in the last 5 seconds (%d total):' % rdd.count())
    for tuple in top_list:
        print('%s (%d happiness)' % (tuple[1], tuple[0]))
if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: network_wordjoinsentiments.py <hostname> <port>', file=sys.stderr)
        sys.exit(-1)
    sc = SparkContext(appName='PythonStreamingNetworkWordJoinSentiments')
    ssc = StreamingContext(sc, 5)

    def line_to_tuple(line: str) -> Tuple[str, str]:
        if False:
            for i in range(10):
                print('nop')
        try:
            (k, v) = line.split(' ')
            return (k, v)
        except ValueError:
            return ('', '')
    word_sentiments_file_path = 'data/streaming/AFINN-111.txt'
    word_sentiments = ssc.sparkContext.textFile(word_sentiments_file_path).map(line_to_tuple)
    lines = ssc.socketTextStream(sys.argv[1], int(sys.argv[2]))
    word_counts = lines.flatMap(lambda line: line.split(' ')).map(lambda word: (word, 1)).reduceByKey(lambda a, b: a + b)
    happiest_words: DStream[Tuple[float, str]] = word_counts.transform(lambda rdd: word_sentiments.join(rdd)).map(lambda word_tuples: (word_tuples[0], float(word_tuples[1][0]) * word_tuples[1][1])).map(lambda word_happiness: (word_happiness[1], word_happiness[0])).transform(lambda rdd: rdd.sortByKey(False))
    happiest_words.foreachRDD(print_happiest_words)
    ssc.start()
    ssc.awaitTermination()