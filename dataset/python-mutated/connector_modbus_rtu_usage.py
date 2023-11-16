"""
Purpose

Shows how to implement an AWS Lambda function that publishes messages to an
AWS IoT Greengrass connector.
"""
import json
import greengrasssdk
iot_client = greengrasssdk.client('iot-data')
send_topic = 'modbus/adapter/request'

def create_read_coils_request():
    if False:
        return 10
    return {'request': {'operation': 'ReadCoilsRequest', 'device': 1, 'address': 1, 'count': 1}, 'id': 'TestRequest'}

def publish_basic_message():
    if False:
        print('Hello World!')
    iot_client.publish(topic=send_topic, payload=json.dumps(create_read_coils_request()))
publish_basic_message()

def function_handler(event, context):
    if False:
        return 10
    return