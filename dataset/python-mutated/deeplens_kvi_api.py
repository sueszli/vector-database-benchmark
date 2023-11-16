import time
import os
import DeepLens_Kinesis_Video as dkv
from botocore.session import Session
import greengrasssdk

def greengrass_hello_world_run():
    if False:
        print('Hello World!')
    client = greengrasssdk.client('iot-data')
    iot_topic = '$aws/things/{}/infer'.format(os.environ['AWS_IOT_THING_NAME'])
    stream_name = 'myStream'
    retention = 2
    wait_time = 60 * 60 * 5
    session = Session()
    creds = session.get_credentials()
    producer = dkv.createProducer(creds.access_key, creds.secret_key, creds.token, 'us-east-1')
    client.publish(topic=iot_topic, payload='Producer created')
    kvs_stream = producer.createStream(stream_name, retention)
    client.publish(topic=iot_topic, payload='Stream {} created'.format(stream_name))
    kvs_stream.start()
    client.publish(topic=iot_topic, payload='Stream started')
    time.sleep(wait_time)
    kvs_stream.stop()
    client.publish(topic=iot_topic, payload='Stream stopped')
greengrass_hello_world_run()