import logging

def some_function():
    if False:
        print('Hello World!')
    logging.info('Some message')

class SomeClass:

    def __init__(self):
        if False:
            return 10
        pass

    def some_method(self):
        if False:
            for i in range(10):
                print('nop')
        pass