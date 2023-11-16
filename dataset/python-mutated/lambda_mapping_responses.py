def handler(event, context):
    if False:
        return 10
    body = event.get('body')
    status_code = body.get('httpStatus')
    if int(status_code) >= 400:
        return {'statusCode': 200, 'body': 'customerror'}
    else:
        return {'statusCode': 200, 'body': 'noerror'}