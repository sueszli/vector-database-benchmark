from setuptools_rust_starter import ExampleClass, PythonClass

def test_python_class() -> None:
    if False:
        return 10
    py_class = PythonClass(value=10)
    assert py_class.value == 10

def test_example_class() -> None:
    if False:
        for i in range(10):
            print('nop')
    example = ExampleClass(value=11)
    assert example.value == 11

def test_doc() -> None:
    if False:
        for i in range(10):
            print('nop')
    import setuptools_rust_starter
    assert setuptools_rust_starter.__doc__ == 'An example module implemented in Rust using PyO3.'