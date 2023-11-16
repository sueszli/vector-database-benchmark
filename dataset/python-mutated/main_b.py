import json

def handler(event, context):
    if False:
        print('Hello World!')
    '\n    FunctionB in child template\n    '
    return {'statusCode': 200, 'body': json.dumps({'hello': 'world'})}