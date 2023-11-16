import json
import os
trace_id_outside_handler = str(os.environ.get('_X_AMZN_TRACE_ID'))

def handler(event, context):
    if False:
        while True:
            i = 10
    response = {'trace_id_outside_handler': trace_id_outside_handler, 'trace_id_inside_handler': str(os.environ.get('_X_AMZN_TRACE_ID'))}
    print(json.dumps(response))
    return response