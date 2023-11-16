import settings
constant = settings.SETTING1

def handler(event, context):
    if False:
        for i in range(10):
            print('nop')
    return constant