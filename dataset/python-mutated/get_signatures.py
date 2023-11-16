import inspect
import warnings
import black
from sphinx.util.inspect import signature
from sphinx.util.inspect import stringify_signature
from . import utils

def get_signature_start(function):
    if False:
        print('Hello World!')
    "For the Dense layer, it should return the string 'keras.layers.Dense'"
    if utils.ismethod(function):
        prefix = f'{utils.get_class_from_method(function).__name__}.'
    else:
        try:
            prefix = f'{function.__module__}.'
        except AttributeError:
            warnings.warn(f'function {function} has no module. It will not be included in the signature.')
            prefix = ''
    return f'{prefix}{function.__name__}'

def get_signature_end(function):
    if False:
        while True:
            i = 10
    sig = signature(function)
    signature_end = stringify_signature(sig, show_annotation=False)
    if utils.ismethod(function):
        signature_end = signature_end.replace('(self, ', '(')
        signature_end = signature_end.replace('(self)', '()')
    return signature_end

def get_function_signature(function, override=None, max_line_length: int=110):
    if False:
        i = 10
        return i + 15
    if override is None:
        signature_start = get_signature_start(function)
    else:
        signature_start = override
    signature_end = get_signature_end(function)
    return format_signature(signature_start, signature_end, max_line_length)

def get_class_signature(cls, override=None, max_line_length: int=110):
    if False:
        print('Hello World!')
    if override is None:
        signature_start = f'{cls.__module__}.{cls.__name__}'
    else:
        signature_start = override
    signature_end = get_signature_end(cls.__init__)
    return format_signature(signature_start, signature_end, max_line_length)

def get_signature(object_, override=None, max_line_length: int=110):
    if False:
        while True:
            i = 10
    if inspect.isclass(object_):
        return get_class_signature(object_, override, max_line_length)
    else:
        return get_function_signature(object_, override, max_line_length)

def format_signature(signature_start: str, signature_end: str, max_line_length: int=110):
    if False:
        return 10
    'pretty formatting to avoid long signatures on one single line'
    fake_signature_start = 'x' * len(signature_start)
    fake_signature = fake_signature_start + signature_end
    fake_python_code = f'def {fake_signature}:\n    pass\n'
    mode = black.FileMode(line_length=max_line_length)
    formatted_fake_python_code = black.format_str(fake_python_code, mode=mode)
    new_signature_end = extract_signature_end(formatted_fake_python_code)
    return signature_start + new_signature_end

def extract_signature_end(function_definition):
    if False:
        print('Hello World!')
    start = function_definition.find('(')
    stop = function_definition.rfind(')')
    return function_definition[start:stop + 1]