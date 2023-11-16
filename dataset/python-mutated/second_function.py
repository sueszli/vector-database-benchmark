import requests

def handler(event, context):
    if False:
        print('Hello World!')
    print(requests.__version__)
    return 'Hello Mars'