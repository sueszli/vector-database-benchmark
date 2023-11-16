import os

def handler(event, context):
    if False:
        return 10
    init_type = os.environ['AWS_LAMBDA_INITIALIZATION_TYPE']
    print(f'init_type={init_type!r}')
    return init_type