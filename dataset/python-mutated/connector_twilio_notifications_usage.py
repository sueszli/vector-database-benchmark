"""
Purpose

Shows how to implement an AWS Lambda function that publishes messages to an
AWS IoT Greengrass connector.
"""
import json
import greengrasssdk
iot_client = greengrasssdk.client('iot-data')
txt_input_topic = 'twilio/txt'

def publish_basic_message():
    if False:
        i = 10
        return i + 15
    message = {'request': {'recipient': {'name': 'Darla', 'phone_number': '+12345000000', 'message': 'Hello from the edge'}, 'from_number': '+19999999999'}, 'id': 'request123'}
    print(f'Message to publish: {message}')
    iot_client.publish(topic=txt_input_topic, payload=json.dumps(message))
publish_basic_message()

def function_handler(event, context):
    if False:
        print('Hello World!')
    return