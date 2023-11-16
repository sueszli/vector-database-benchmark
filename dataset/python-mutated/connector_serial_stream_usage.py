"""
Purpose

Shows how to implement an AWS Lambda function that publishes messages to an
AWS IoT Greengrass connector.
"""
import json
import greengrasssdk
iot_client = greengrasssdk.client('iot-data')
send_topic = 'serial/CORE_THING_NAME/write/dev/serial1'

def create_serial_stream_request():
    if False:
        i = 10
        return i + 15
    return {'data': 'TEST', 'type': 'ascii', 'id': 'abc123'}

def publish_basic_message():
    if False:
        for i in range(10):
            print('nop')
    iot_client.publish(topic=send_topic, payload=json.dumps(create_serial_stream_request()))
publish_basic_message()

def function_handler(event, context):
    if False:
        print('Hello World!')
    return