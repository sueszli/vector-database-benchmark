"""Fake HUG API module usable for testing importation of modules"""
import hug

class FakeException(BaseException):
    pass

@hug.directive(apply_globally=False)
def my_directive(default=None, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    'for testing'
    return default

@hug.default_input_format('application/made-up')
def made_up_formatter(data):
    if False:
        print('Hello World!')
    'for testing'
    return data

@hug.default_output_format()
def output_formatter(data):
    if False:
        i = 10
        return i + 15
    'for testing'
    return hug.output_format.json(data)

@hug.get()
def made_up_api(hug_my_directive=True):
    if False:
        i = 10
        return i + 15
    'for testing'
    return hug_my_directive

@hug.directive(apply_globally=True)
def my_directive_global(default=None, **kwargs):
    if False:
        return 10
    'for testing'
    return default

@hug.default_input_format('application/made-up', apply_globally=True)
def made_up_formatter_global(data):
    if False:
        print('Hello World!')
    'for testing'
    return data

@hug.default_output_format(apply_globally=True)
def output_formatter_global(data, request=None, response=None):
    if False:
        while True:
            i = 10
    'for testing'
    return hug.output_format.json(data)

@hug.request_middleware()
def handle_request(request, response):
    if False:
        for i in range(10):
            print('nop')
    'for testing'
    return

@hug.startup()
def on_startup(api):
    if False:
        print('Hello World!')
    'for testing'
    return

@hug.static()
def static():
    if False:
        return 10
    'for testing'
    return ('',)

@hug.sink('/all')
def sink(path):
    if False:
        for i in range(10):
            print('nop')
    'for testing'
    return path

@hug.exception(FakeException)
def handle_exception(exception):
    if False:
        print('Hello World!')
    'Handles the provided exception for testing'
    return True

@hug.not_found()
def not_found_handler():
    if False:
        print('Hello World!')
    'for testing'
    return True