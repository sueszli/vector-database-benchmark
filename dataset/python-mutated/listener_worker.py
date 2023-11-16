"""
A worker for streaming query listener in Spark Connect.
Usually this is ran on the driver side of the Spark Connect Server.
"""
import os
import json
from pyspark.java_gateway import local_connect_and_auth
from pyspark.serializers import read_int, write_int, UTF8Deserializer, CPickleSerializer
from pyspark import worker
from pyspark.sql import SparkSession
from pyspark.util import handle_worker_exception
from typing import IO
from pyspark.sql.streaming.listener import QueryStartedEvent, QueryProgressEvent, QueryTerminatedEvent, QueryIdleEvent
from pyspark.worker_util import check_python_version
pickle_ser = CPickleSerializer()
utf8_deserializer = UTF8Deserializer()

def main(infile: IO, outfile: IO) -> None:
    if False:
        return 10
    check_python_version(infile)
    connect_url = os.environ['SPARK_CONNECT_LOCAL_URL']
    session_id = utf8_deserializer.loads(infile)
    print(f'Streaming query listener worker is starting with url {connect_url} and sessionId {session_id}.')
    spark_connect_session = SparkSession.builder.remote(connect_url).getOrCreate()
    spark_connect_session._client._session_id = session_id
    listener = worker.read_command(pickle_ser, infile)
    write_int(0, outfile)
    outfile.flush()
    listener._set_spark_session(spark_connect_session)
    assert listener.spark == spark_connect_session

    def process(listener_event_str, listener_event_type):
        if False:
            i = 10
            return i + 15
        listener_event = json.loads(listener_event_str)
        if listener_event_type == 0:
            listener.onQueryStarted(QueryStartedEvent.fromJson(listener_event))
        elif listener_event_type == 1:
            listener.onQueryProgress(QueryProgressEvent.fromJson(listener_event))
        elif listener_event_type == 2:
            listener.onQueryIdle(QueryIdleEvent.fromJson(listener_event))
        elif listener_event_type == 3:
            listener.onQueryTerminated(QueryTerminatedEvent.fromJson(listener_event))
    while True:
        event = utf8_deserializer.loads(infile)
        event_type = read_int(infile)
        try:
            process(event, int(event_type))
            write_int(0, outfile)
        except BaseException as e:
            handle_worker_exception(e, outfile)
        outfile.flush()
if __name__ == '__main__':
    java_port = int(os.environ['PYTHON_WORKER_FACTORY_PORT'])
    auth_secret = os.environ['PYTHON_WORKER_FACTORY_SECRET']
    (sock_file, sock) = local_connect_and_auth(java_port, auth_secret)
    sock.settimeout(None)
    main(sock_file, sock_file)