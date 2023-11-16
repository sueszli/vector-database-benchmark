class CustomException(Exception):
    pass

def handler(event, context):
    if False:
        print('Hello World!')
    raise CustomException('some error occurred')