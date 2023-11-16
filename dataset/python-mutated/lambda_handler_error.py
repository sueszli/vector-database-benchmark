def handler(event, context):
    if False:
        for i in range(10):
            print('nop')
    raise Exception('Handler fails')