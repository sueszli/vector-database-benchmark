"""
A worker for streaming foreachBatch in Spark Connect.
Usually this is ran on the driver side of the Spark Connect Server.
"""
import os
from pyspark.java_gateway import local_connect_and_auth
from pyspark.serializers import write_int, read_long, UTF8Deserializer, CPickleSerializer
from pyspark import worker
from pyspark.sql import SparkSession
from pyspark.util import handle_worker_exception
from typing import IO
from pyspark.worker_util import check_python_version
pickle_ser = CPickleSerializer()
utf8_deserializer = UTF8Deserializer()

def main(infile: IO, outfile: IO) -> None:
    if False:
        for i in range(10):
            print('nop')
    check_python_version(infile)
    connect_url = os.environ['SPARK_CONNECT_LOCAL_URL']
    session_id = utf8_deserializer.loads(infile)
    print(f'Streaming foreachBatch worker is starting with url {connect_url} and sessionId {session_id}.')
    spark_connect_session = SparkSession.builder.remote(connect_url).getOrCreate()
    spark_connect_session._client._session_id = session_id
    func = worker.read_command(pickle_ser, infile)
    write_int(0, outfile)
    outfile.flush()
    log_name = 'Streaming ForeachBatch worker'

    def process(df_id, batch_id):
        if False:
            for i in range(10):
                print('nop')
        print(f'{log_name} Started batch {batch_id} with DF id {df_id}')
        batch_df = spark_connect_session._create_remote_dataframe(df_id)
        func(batch_df, batch_id)
        print(f'{log_name} Completed batch {batch_id} with DF id {df_id}')
    while True:
        df_ref_id = utf8_deserializer.loads(infile)
        batch_id = read_long(infile)
        try:
            process(df_ref_id, int(batch_id))
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