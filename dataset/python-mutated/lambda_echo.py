import json

def handler(event, context):
    if False:
        i = 10
        return i + 15
    print(json.dumps(event))
    return event