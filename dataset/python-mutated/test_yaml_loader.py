from readthedocs.doc_builder.backends.mkdocs import ProxyPythonName, yaml_dump_safely, yaml_load_safely
content = '\nint: 3\nfloat: !!float 3\nfunction: !!python/name:python_function\nother_function: !!python/name:module.other.function\nunknown: !!python/module:python_module\n'

def test_yaml_load_safely():
    if False:
        for i in range(10):
            print('nop')
    expected = {'int': 3, 'float': 3.0, 'function': ProxyPythonName('python_function'), 'other_function': ProxyPythonName('module.other.function'), 'unknown': None}
    data = yaml_load_safely(content)
    assert data == expected
    assert type(data['int']) is int
    assert type(data['float']) is float
    assert type(data['function']) is ProxyPythonName
    assert type(data['other_function']) is ProxyPythonName
    assert data['function'].value == 'python_function'
    assert data['other_function'].value == 'module.other.function'

def test_yaml_dump_safely():
    if False:
        while True:
            i = 10
    data = yaml_load_safely(content)
    assert yaml_load_safely(yaml_dump_safely(data)) == data