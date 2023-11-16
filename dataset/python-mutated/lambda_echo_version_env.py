import json
import os

def handler(event, context):
    if False:
        i = 10
        return i + 15
    print(json.dumps({'function_version': os.environ.get('AWS_LAMBDA_FUNCTION_VERSION'), 'CUSTOM_VAR': os.environ.get('CUSTOM_VAR'), 'event': event}))
    return event