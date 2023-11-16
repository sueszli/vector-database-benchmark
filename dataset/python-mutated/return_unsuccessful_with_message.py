def handler(event, context):
    if False:
        return 10
    template = event['fragment']
    return {'requestId': event['requestId'], 'status': 'failed', 'fragment': template, 'errorMessage': 'failed because it is a test'}