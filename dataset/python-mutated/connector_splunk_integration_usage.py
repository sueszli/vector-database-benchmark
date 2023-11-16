"""
Purpose

Shows how to implement an AWS Lambda function that publishes messages to an
AWS IoT Greengrass connector.
"""
import json
import greengrasssdk
iot_client = greengrasssdk.client('iot-data')
send_topic = 'splunk/logs/put'

def create_request():
    if False:
        for i in range(10):
            print('nop')
    return {'request': {'event': 'Access log test message.'}, 'id': 'req_123'}

def publish_basic_message():
    if False:
        print('Hello World!')
    message = create_request()
    print(f'Message to publish: {message}')
    iot_client.publish(topic=send_topic, payload=json.dumps(message))
publish_basic_message()

def function_handler(event, context):
    if False:
        return 10
    return