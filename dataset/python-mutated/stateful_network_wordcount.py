"""
 Counts words in UTF8 encoded, '\\n' delimited text received from the
 network every second.

 Usage: stateful_network_wordcount.py <hostname> <port>
   <hostname> and <port> describe the TCP server that Spark Streaming
    would connect to receive data.

 To run this on your local machine, you need to first run a Netcat server
    `$ nc -lk 9999`
 and then run the example
    `$ bin/spark-submit examples/src/main/python/streaming/stateful_network_wordcount.py \\
        localhost 9999`
"""
import sys
from typing import Iterable, Optional
from pyspark import SparkContext
from pyspark.streaming import StreamingContext
if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: stateful_network_wordcount.py <hostname> <port>', file=sys.stderr)
        sys.exit(-1)
    sc = SparkContext(appName='PythonStreamingStatefulNetworkWordCount')
    ssc = StreamingContext(sc, 1)
    ssc.checkpoint('checkpoint')
    initialStateRDD = sc.parallelize([(u'hello', 1), (u'world', 1)])

    def updateFunc(new_values: Iterable[int], last_sum: Optional[int]) -> int:
        if False:
            return 10
        return sum(new_values) + (last_sum or 0)
    lines = ssc.socketTextStream(sys.argv[1], int(sys.argv[2]))
    running_counts = lines.flatMap(lambda line: line.split(' ')).map(lambda word: (word, 1)).updateStateByKey(updateFunc, initialRDD=initialStateRDD)
    running_counts.pprint()
    ssc.start()
    ssc.awaitTermination()