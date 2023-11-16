"""Simple 1 endpoint Fake HUG API module usable for testing importation of modules"""
import hug

class FakeSimpleException(Exception):
    pass

@hug.get()
def made_up_hello():
    if False:
        print('Hello World!')
    'for science!'
    return 'hello'

@hug.get('/exception')
def made_up_exception():
    if False:
        print('Hello World!')
    raise FakeSimpleException('test')