import logging

class PassRegistry:

    def __init__(self):
        if False:
            print('Hello World!')
        self.passes = {}

    def __getitem__(self, pass_id):
        if False:
            i = 10
            return i + 15
        "\n        pass_id (str): namespace::func_name (e.g., 'common::const_elimination')\n        "
        if pass_id not in self.passes:
            raise KeyError('Pass {} not found'.format(pass_id))
        return self.passes[pass_id]

    def add(self, namespace, pass_func):
        if False:
            print('Hello World!')
        func_name = pass_func.__name__
        pass_id = namespace + '::' + func_name
        logging.debug('Registering pass {}'.format(pass_id))
        if pass_id in self.passes:
            msg = 'Pass {} already registered.'
            raise KeyError(msg.format(pass_id))
        self.passes[pass_id] = pass_func
PASS_REGISTRY = PassRegistry()

def register_pass(namespace):
    if False:
        print('Hello World!')
    "\n    namespaces like {'common', 'nn_backend', <other-backends>,\n    <other-frontends>}\n    "

    def func_wrapper(pass_func):
        if False:
            for i in range(10):
                print('nop')
        PASS_REGISTRY.add(namespace, pass_func)
        return pass_func
    return func_wrapper