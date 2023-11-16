import json
import os
import boto3

def handler(event, context):
    if False:
        for i in range(10):
            print('nop')
    endpoint_url = None
    if os.environ.get('AWS_ENDPOINT_URL'):
        endpoint_url = os.environ['AWS_ENDPOINT_URL']
    sf = boto3.client('stepfunctions', endpoint_url=endpoint_url, region_name=event['region_name'], verify=False)
    sf.start_execution(stateMachineArn=event['state_machine_arn'], input=json.dumps(event['input']))
    return 0