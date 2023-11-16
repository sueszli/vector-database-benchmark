"""
Purpose

Shows how to implement an AWS Lambda function that publishes messages to an
AWS IoT Greengrass connector.
"""
import json
import random
import greengrasssdk
iot_client = greengrasssdk.client('iot-data')
send_topic = 'twilio/txt'

def create_request(event):
    if False:
        return 10
    return {'request': {'recipient': {'name': event['to_name'], 'phone_number': event['to_number'], 'message': f"temperature:{event['temperature']}"}}, 'id': f'request_{random.randint(1, 101)}'}

def function_handler(event, context):
    if False:
        return 10
    temperature = event['temperature']
    if temperature > 30:
        message = create_request(event)
        iot_client.publish(topic='twilio/txt', payload=json.dumps(message))
        print(f'Published: {message}')
    print(f'Temperature: {temperature}')