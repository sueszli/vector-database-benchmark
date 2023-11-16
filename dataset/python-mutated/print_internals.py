import json

def handler(event, context):
    if False:
        while True:
            i = 10
    fragment = event['fragment']
    fragment['Resources']['Parameter']['Properties']['Value'] = json.dumps({'Event': event})
    return {'requestId': event['requestId'], 'status': 'success', 'fragment': fragment, 'errorMessage': 'test-error message'}