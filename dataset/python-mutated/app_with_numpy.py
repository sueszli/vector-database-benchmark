import json
import numpy as np

def lambda_handler(event, context):
    if False:
        for i in range(10):
            print('nop')
    return {'statusCode': 200, 'body': json.dumps({'message': 'hello mars', 'extra_message': np.array([1, 2, 3, 4, 5, 6]).tolist()})}