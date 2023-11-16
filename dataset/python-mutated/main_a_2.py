import json

def handler(event, context):
    if False:
        for i in range(10):
            print('nop')
    '\n    FunctionA in leaf template\n    '
    return {'statusCode': 200, 'body': json.dumps({'hello': 'world'})}