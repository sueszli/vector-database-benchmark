import json

def handler(event, context):
    if False:
        while True:
            i = 10
    print(json.dumps({'event': event, 'aws_request_id': context.aws_request_id}))
    raise Exception('intentional failure')