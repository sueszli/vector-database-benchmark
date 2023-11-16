"""
Purpose

Shows how to implement an AWS Lambda function that publishes messages to an
AWS IoT Greengrass connector.
"""
import json
import greengrasssdk
iot_client = greengrasssdk.client('iot-data')
send_topic = 'servicenow/metricbase/metric'

def create_request():
    if False:
        while True:
            i = 10
    return {'request': {'subject': '2efdf6badbd523803acfae441b961961', 'metric_name': 'u_count', 'value': 1234, 'timestamp': '2018-10-20T20:22:20', 'table': 'u_greengrass_metricbase_test'}}

def publish_basic_message():
    if False:
        for i in range(10):
            print('nop')
    message = create_request()
    print(f'Message to publish: {message}')
    iot_client.publish(topic=send_topic, payload=json.dumps(message))
publish_basic_message()

def function_handler(event, context):
    if False:
        return 10
    return