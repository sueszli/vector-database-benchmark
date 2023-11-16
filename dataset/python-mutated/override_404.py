import hug

@hug.get()
def hello_world():
    if False:
        print('Hello World!')
    return 'Hello world!'

@hug.not_found()
def not_found():
    if False:
        for i in range(10):
            print('nop')
    return {'Nothing': 'to see'}