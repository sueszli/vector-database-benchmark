class PythonClass:
    python_string = 'hello'
    python_integer = None
    LIST__python_list = ['a', 'b', 'c']

    def __init__(self):
        if False:
            print('Hello World!')
        self.python_integer = 42

    def python_method(self):
        if False:
            i = 10
            return i + 15
        pass

    @property
    def python_property(self):
        if False:
            return 10
        return 'value'