"""
 Split lines into words, group by words and use the state per key to track session of each key.
 Each session window sets a 10 seconds processing time timeout.
 After 10 seconds of idle period, the session summary will be finalized and output to sink.
 Usage: structured_network_wordcount_windowed.py <hostname> <port>
 <hostname> and <port> describe the TCP server that Structured Streaming
 would connect to receive data.

 To run this on your local machine, you need to first run a Netcat server
    `$ nc -lk 9999`
 and then run the example
    `$ bin/spark-submit
    examples/src/main/python/sql/streaming/structured_network_wordcount_session_window.py
    localhost 9999`
"""
import sys
from typing import Iterable, Any
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import explode
from pyspark.sql.functions import split
from pyspark.sql.types import LongType, StringType, TimestampType, StructType, StructField
from pyspark.sql.streaming.state import GroupStateTimeout, GroupState
if __name__ == '__main__':
    if len(sys.argv) != 3:
        msg = 'Usage: structured_network_wordcount_session_window.py <hostname> <port>'
        print(msg, file=sys.stderr)
        sys.exit(-1)
    host = sys.argv[1]
    port = int(sys.argv[2])
    spark = SparkSession.builder.appName('StructuredNetworkWordCountSessionWindow').getOrCreate()
    lines = spark.readStream.format('socket').option('host', host).option('port', port).option('includeTimestamp', 'true').load()
    events = lines.select(explode(split(lines.value, ' ')).alias('sessionId'), lines.timestamp)
    session_schema = StructType([StructField('sessionId', StringType()), StructField('count', LongType()), StructField('start', TimestampType()), StructField('end', TimestampType())])
    session_state_schema = StructType([StructField('count', LongType()), StructField('start', TimestampType()), StructField('end', TimestampType())])

    def func(key: Any, pdfs: Iterable[pd.DataFrame], state: GroupState) -> Iterable[pd.DataFrame]:
        if False:
            return 10
        if state.hasTimedOut:
            (count, start, end) = state.get
            state.remove()
            (session_id,) = key
            yield pd.DataFrame({'sessionId': [session_id], 'count': [count], 'start': [start], 'end': [end]})
        else:
            pdf_iter = iter(pdfs)
            first_pdf = next(pdf_iter)
            start = first_pdf['timestamp'].min()
            end = first_pdf['timestamp'].max()
            count = len(first_pdf)
            for pdf in pdf_iter:
                start = min(start, pdf['timestamp'].min())
                end = max(end, pdf['timestamp'].max())
                count = count + len(pdf)
            if state.exists:
                (old_count, start, old_end) = state.get
                count = count + old_count
                end = max(end, old_end)
            state.update((count, start, end))
            state.setTimeoutDuration(10000)
            yield pd.DataFrame()
    sessions = events.groupBy(events['sessionId']).applyInPandasWithState(func, session_schema, session_state_schema, 'Update', GroupStateTimeout.ProcessingTimeTimeout)
    query = sessions.writeStream.outputMode('update').format('console').start()
    query.awaitTermination()