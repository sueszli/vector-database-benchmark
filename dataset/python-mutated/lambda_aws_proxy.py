import json

def handler(event, context):
    if False:
        for i in range(10):
            print('nop')
    print(json.dumps(event))
    return {'statusCode': 200, 'body': json.dumps(event), 'isBase64Encoded': False}