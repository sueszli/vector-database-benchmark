import json

def handler(event, context):
    if False:
        i = 10
        return i + 15
    return {'isBase64Encoded': False, 'headers': {}, 'body': json.dumps({'test': 'hello world'}), 'statusCode': 200}