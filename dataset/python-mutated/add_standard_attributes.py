def handler(event, context):
    if False:
        for i in range(10):
            print('nop')
    fragment = add_standard_attributes(event['fragment'])
    return {'requestId': event['requestId'], 'status': 'success', 'fragment': fragment}

def add_standard_attributes(fragment):
    if False:
        print('Hello World!')
    fragment['FifoTopic'] = True
    fragment['ContentBasedDeduplication'] = True
    return fragment