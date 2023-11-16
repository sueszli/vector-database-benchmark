import json

def lambda_handler(event, context):
    if False:
        i = 10
        return i + 15
    return {'statusCode': 200, 'body': json.dumps({'message': 'hello mars'})}