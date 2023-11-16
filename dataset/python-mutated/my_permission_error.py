from typing import Type

def get_permission_error_class():
    if False:
        for i in range(10):
            print('nop')
    try:
        return PermissionError
    except NameError:
        return OSError
MyPermissionError = get_permission_error_class()