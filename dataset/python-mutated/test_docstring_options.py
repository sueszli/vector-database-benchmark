def test_docstring_options():
    if False:
        print('Hello World!')
    from pybind11_tests import test_function1, test_function2, test_function3, test_function4, test_function5, test_function6, test_function7, DocstringTestFoo
    assert not test_function1.__doc__
    assert test_function2.__doc__ == 'A custom docstring'
    assert test_function3.__doc__.startswith('test_function3(a: int, b: int) -> None')
    assert test_function4.__doc__.startswith('test_function4(a: int, b: int) -> None')
    assert test_function4.__doc__.endswith('A custom docstring\n')
    assert not test_function5.__doc__
    assert test_function6.__doc__ == 'A custom docstring'
    assert test_function7.__doc__.startswith('test_function7(a: int, b: int) -> None')
    assert test_function7.__doc__.endswith('A custom docstring\n')
    assert not DocstringTestFoo.__doc__
    assert not DocstringTestFoo.value_prop.__doc__