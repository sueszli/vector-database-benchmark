import sys

def get_variables():
    if False:
        for i in range(10):
            print('nop')
    variables = {'integer': 42, 'float': 3.14, 'byte_string': b'hyv\xe4', 'byte_string_str': 'hyv\\xe4', 'boolean': True, 'none': None, 'module': sys, 'module_str': str(sys), 'list': [1, b'\xe4', 'ä'], 'dict': {b'\xe4': 'ä'}, 'list_str': u"[1, b'\\xe4', 'ä']", 'dict_str': u"{b'\\xe4': 'ä'}"}
    return variables