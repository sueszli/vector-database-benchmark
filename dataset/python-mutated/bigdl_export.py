import functools
import sys
Keras_API_NAME = 'keras'

class api_export(object):

    def __init__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        'Export under the names *args (first one is considered canonical).\n\n    Args:\n      *args: API names in dot delimited format.\n      **kwargs: Optional keyed arguments.\n        api_name: Name of the API you want to generate (e.g. `tensorflow` or\n          `estimator`). Default is `keras`.\n    '
        self._names = args
        self._api_name = kwargs.get('api_name', Keras_API_NAME)

    def __call__(self, func):
        if False:
            i = 10
            return i + 15
        for name in self._names:
            sys.modules[name] = func
        return func
keras_export = functools.partial(api_export, api_name=Keras_API_NAME)