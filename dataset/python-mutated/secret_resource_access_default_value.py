"""
Purpose

Shows how to implement an AWS Lambda function that uses AWS IoT Greengrass Core SDK
to get a secret.
"""
import greengrasssdk
secrets_client = greengrasssdk.client('secretsmanager')
secret_name = 'greengrass-MySecret-abc'

def function_handler(event, context):
    if False:
        while True:
            i = 10
    response = secrets_client.get_secret_value(SecretId=secret_name)
    secret = response.get('SecretString')