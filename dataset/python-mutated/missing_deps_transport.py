"""Transport that has missing deps"""
import io
try:
    import this_module_does_not_exist_but_we_need_it
except ImportError:
    MISSING_DEPS = True
SCHEME = 'missing'
open = io.open

def parse_uri(uri_as_string):
    if False:
        print('Hello World!')
    ...

def open_uri(uri_as_string, mode, transport_params):
    if False:
        for i in range(10):
            print('nop')
    ...