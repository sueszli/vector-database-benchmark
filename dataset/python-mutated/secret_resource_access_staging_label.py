"""
Purpose

Shows how to implement an AWS Lambda function that uses AWS IoT Greengrass Core SDK
to get the specific version of a secret.
"""
import greengrasssdk
secrets_client = greengrasssdk.client('secretsmanager')
secret_name = 'greengrass-TestSecret'
secret_version = 'MyTargetLabel'

def function_handler(event, context):
    if False:
        i = 10
        return i + 15
    response = secrets_client.get_secret_value(SecretId=secret_name, VersionStage=secret_version)
    secret = response.get('SecretString')