from ExampleLibrary import ExampleLibrary

class ExtendPythonLib(ExampleLibrary):

    def kw_in_python_extender(self, arg):
        if False:
            while True:
                i = 10
        return arg / 2

    def print_many(self, *msgs):
        if False:
            while True:
                i = 10
        raise Exception('Overridden kw executed!')

    def using_method_from_python_parent(self):
        if False:
            return 10
        self.exception('AssertionError', 'Error message from lib')