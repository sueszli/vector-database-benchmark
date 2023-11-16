"""
Purpose

Shows how to implement an AWS Lambda function that publishes messages to an
AWS IoT Greengrass connector.
"""
import json
import greengrasssdk
iot_client = greengrasssdk.client('iot-data')
send_topic = 'sns/message'

def create_request():
    if False:
        return 10
    return {'request': {'message': 'Message from SNS Connector Test'}, 'id': 'req_123'}

def publish_basic_message():
    if False:
        print('Hello World!')
    message = create_request()
    print(f'Message to publish: {message}')
    iot_client.publish(topic=send_topic, payload=json.dumps(message))
publish_basic_message()

def function_handler(event, context):
    if False:
        for i in range(10):
            print('nop')
    pass