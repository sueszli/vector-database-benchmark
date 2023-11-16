import os
import uuid
import boto3

def handler(event, context):
    if False:
        i = 10
        return i + 15
    r = event['Records'][0]
    region = r['awsRegion']
    s3_metadata = r['s3']['object']
    table_name = s3_metadata['key']
    endpoint_url = None
    if os.environ.get('AWS_ENDPOINT_URL'):
        endpoint_url = os.environ['AWS_ENDPOINT_URL']
    ddb = boto3.resource('dynamodb', endpoint_url=endpoint_url, region_name=region, verify=False)
    ddb.Table(table_name).put_item(Item={'uuid': str(uuid.uuid4())[0:8], 'data': r})