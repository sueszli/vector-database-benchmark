import json
from root_layer import layer_method
import requests

def lambda_handler(event, context):
    if False:
        while True:
            i = 10
    'Sample pure Lambda function that returns a message and a location'
    try:
        ip = requests.get('http://checkip.amazonaws.com/')
    except requests.RequestException as e:
        print(e)
        raise e
    return {'statusCode': 200, 'body': json.dumps({'message': f'{layer_method() + 6}', 'location': ip.text.replace('\n', '')})}