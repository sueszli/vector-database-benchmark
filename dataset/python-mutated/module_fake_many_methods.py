"""Fake HUG API module usable for testing importation of modules"""
import hug

@hug.get()
def made_up_hello():
    if False:
        for i in range(10):
            print('nop')
    'GETting for science!'
    return 'hello from GET'

@hug.post()
def made_up_hello():
    if False:
        print('Hello World!')
    'POSTing for science!'
    return 'hello from POST'