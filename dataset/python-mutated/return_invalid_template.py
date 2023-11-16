def handler(event, context):
    if False:
        print('Hello World!')
    return {'requestId': event['requestId'], 'status': 'success', 'fragment': 'invalid'}