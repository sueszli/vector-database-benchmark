"""Test: imports within `ModuleNotFoundError` and `ImportError` handlers."""

def module_not_found_error():
    if False:
        print('Hello World!')
    try:
        import orjson
        return True
    except ModuleNotFoundError:
        return False

def import_error():
    if False:
        while True:
            i = 10
    try:
        import orjson
        return True
    except ImportError:
        return False