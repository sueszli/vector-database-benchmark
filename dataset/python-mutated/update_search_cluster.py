import os
import boto3
import requests
from requests_aws4auth import AWS4Auth
region = os.environ['REGION']
service = 'es'
credentials = boto3.Session().get_credentials()
awsauth = AWS4Auth(credentials.access_key, credentials.secret_key, region, service, session_token=credentials.token)
if os.getenv('LOCALSTACK_HOSTNAME'):
    host = 'http://' + os.environ['ESENDPOINT']
    host = host.replace('localhost.localstack.cloud', os.getenv('LOCALSTACK_HOSTNAME'))
else:
    host = 'https://' + os.environ['ESENDPOINT']
index = 'lambda-index'
type = '_doc'
url = host + '/' + index + '/' + type + '/'
headers = {'Content-Type': 'application/json'}

def handler(event, context):
    if False:
        return 10
    count = 0
    for record in event['Records']:
        id = record['dynamodb']['Keys']['id']['S']
        print('bookId ' + id)
        if record['eventName'] == 'REMOVE':
            requests.delete(url + id, auth=awsauth)
        else:
            document = record['dynamodb']['NewImage']
            print(document)
            r = requests.put(url + id, auth=awsauth, json=document, headers=headers)
            print(r.content)
        count += 1
    print(f'processed {count} records.')
    return str(count) + ' records processed.'