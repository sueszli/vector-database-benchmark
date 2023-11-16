from .utils import Result, requires_pyright, run_pyright, skip_on_windows
pytestmark = [skip_on_windows, requires_pyright]
CODE = '\nimport strawberry\n\n\n@strawberry.type\nclass User:\n    name: str\n    age: strawberry.Private[int]\n\n\npatrick = User(name="Patrick", age=1)\nUser(n="Patrick")\n\nreveal_type(patrick.name)\nreveal_type(patrick.age)\n'

def test_pyright():
    if False:
        while True:
            i = 10
    results = run_pyright(CODE)
    assert results == [Result(type='error', message='No parameter named "n"', line=12, column=6), Result(type='error', message='Arguments missing for parameters "name", "age"', line=12, column=1), Result(type='information', message='Type of "patrick.name" is "str"', line=14, column=13), Result(type='information', message='Type of "patrick.age" is "int"', line=15, column=13)]