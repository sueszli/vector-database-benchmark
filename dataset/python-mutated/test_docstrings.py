from seaborn._docstrings import DocstringComponents
EXAMPLE_DICT = dict(param_a='\na : str\n    The first parameter.\n    ')

class ExampleClass:

    def example_method(self):
        if False:
            while True:
                i = 10
        'An example method.\n\n        Parameters\n        ----------\n        a : str\n           A method parameter.\n\n        '

def example_func():
    if False:
        return 10
    'An example function.\n\n    Parameters\n    ----------\n    a : str\n        A function parameter.\n\n    '

class TestDocstringComponents:

    def test_from_dict(self):
        if False:
            for i in range(10):
                print('nop')
        obj = DocstringComponents(EXAMPLE_DICT)
        assert obj.param_a == 'a : str\n    The first parameter.'

    def test_from_nested_components(self):
        if False:
            for i in range(10):
                print('nop')
        obj_inner = DocstringComponents(EXAMPLE_DICT)
        obj_outer = DocstringComponents.from_nested_components(inner=obj_inner)
        assert obj_outer.inner.param_a == 'a : str\n    The first parameter.'

    def test_from_function(self):
        if False:
            print('Hello World!')
        obj = DocstringComponents.from_function_params(example_func)
        assert obj.a == 'a : str\n    A function parameter.'

    def test_from_method(self):
        if False:
            return 10
        obj = DocstringComponents.from_function_params(ExampleClass.example_method)
        assert obj.a == 'a : str\n    A method parameter.'