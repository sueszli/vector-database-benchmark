def handle(module):
    if False:
        i = 10
        return i + 15
    try:
        module = 1
    except ImportError as exc:
        module = exc
    return module

def handle2(module):
    if False:
        while True:
            i = 10
    if module == 'foo':
        try:
            module = 1
        except ImportError as exc:
            module = exc
    return module
try:
    pass
except ImportError as exc:
    pass
finally:
    y = 1