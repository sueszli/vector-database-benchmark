"""
Purpose

Shows how to implement an AWS Lambda function that uses AWS IoT Greengrass Core SDK
to get a secret.
"""
import greengrasssdk
secrets_client = greengrasssdk.client('secretsmanager')
iot_client = greengrasssdk.client('iot-data')
secret_name = 'greengrass-TestSecret'
send_topic = 'secrets/output'

def function_handler(event, context):
    if False:
        return 10
    '\n    Gets a secret and publishes a message to indicate whether the secret was\n    successfully retrieved.\n    '
    response = secrets_client.get_secret_value(SecretId=secret_name)
    secret_value = response.get('SecretString')
    message = f'Failed to retrieve secret {secret_name}.' if secret_value is None else f'Successfully retrieved secret {secret_name}.'
    iot_client.publish(topic=send_topic, payload=message)
    print('Published: ' + message)