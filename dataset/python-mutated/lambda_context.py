def handler(event, context):
    if False:
        for i in range(10):
            print('nop')
    print(event)
    print(context.aws_request_id)
    if event.get('fail'):
        raise Exception('Intentional failure')
    return context.aws_request_id