import os
import time

def handler(event, context):
    if False:
        i = 10
        return i + 15
    if event.get('sleep'):
        time.sleep(event.get('sleep'))
    return {'environment': dict(os.environ)}