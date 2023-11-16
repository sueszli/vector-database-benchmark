import json
import time

def handler(event, context):
    if False:
        print('Hello World!')
    result = {'executionStart': time.time(), 'event': event}
    time.sleep(5)
    print(json.dumps(result))
    return result