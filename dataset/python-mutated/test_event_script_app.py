from __future__ import print_function

def handler_for_events(event, context):
    if False:
        for i in range(10):
            print('nop')
    print('Event:', event)
    return True