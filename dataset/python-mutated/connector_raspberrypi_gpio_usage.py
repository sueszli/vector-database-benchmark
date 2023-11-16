"""
Purpose

Shows how to implement an AWS Lambda function that publishes messages to an
AWS IoT Greengrass connector.
"""
import os
import greengrasssdk
iot_client = greengrasssdk.client('iot-data')
INPUT_GPIOS = [6, 17, 22]
thing_name = os.environ['AWS_IOT_THING_NAME']

def publish_basic_message():
    if False:
        for i in range(10):
            print('nop')
    for gpio_num in INPUT_GPIOS:
        topic = '/'.join(['gpio', thing_name, str(gpio_num), 'read'])
        iot_client.publish(topic=topic, payload=f'Hello from GPIO {gpio_num}!')
publish_basic_message()

def function_handler(event, context):
    if False:
        return 10
    return