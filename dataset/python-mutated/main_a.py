import json

def handler(event, context):
    if False:
        print('Hello World!')
    '\n    FunctionA in root template\n    '
    return {'statusCode': 200, 'body': json.dumps({'hello': 'world'})}