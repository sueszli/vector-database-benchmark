"""
Parameterized Option Class
"""
import types

def parameterized_option(option):
    if False:
        i = 10
        return i + 15
    'Meta decorator for option decorators.\n    This adds the ability to specify optional parameters for option decorators.\n\n    Usage:\n        @parameterized_option\n        def some_option(f, required=False)\n            ...\n\n        @some_option\n        def command(...)\n\n        or\n\n        @some_option(required=True)\n        def command(...)\n    '

    def parameter_wrapper(*args, **kwargs):
        if False:
            i = 10
            return i + 15
        if len(args) == 1 and isinstance(args[0], types.FunctionType):
            return option(args[0])

        def option_wrapper(f):
            if False:
                print('Hello World!')
            return option(f, *args, **kwargs)
        return option_wrapper
    return parameter_wrapper