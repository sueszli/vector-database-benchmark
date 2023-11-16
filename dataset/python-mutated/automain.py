import sys
from .errors import AutocommandError

class AutomainRequiresModuleError(AutocommandError, TypeError):
    pass

def automain(module, *, args=(), kwargs=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    This decorator automatically invokes a function if the module is being run\n    as the "__main__" module. Optionally, provide args or kwargs with which to\n    call the function. If `module` is "__main__", the function is called, and\n    the program is `sys.exit`ed with the return value. You can also pass `True`\n    to cause the function to be called unconditionally. If the function is not\n    called, it is returned unchanged by the decorator.\n\n    Usage:\n\n    @automain(__name__)  # Pass __name__ to check __name__=="__main__"\n    def main():\n        ...\n\n    If __name__ is "__main__" here, the main function is called, and then\n    sys.exit called with the return value.\n    '
    if callable(module):
        raise AutomainRequiresModuleError(module)
    if module == '__main__' or module is True:
        if kwargs is None:
            kwargs = {}

        def automain_decorator(main):
            if False:
                return 10
            sys.exit(main(*args, **kwargs))
        return automain_decorator
    else:
        return lambda main: main