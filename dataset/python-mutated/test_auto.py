from .utils import Result, requires_pyright, run_pyright, skip_on_windows
pytestmark = [skip_on_windows, requires_pyright]
CODE = '\nimport strawberry\n\n\n@strawberry.type\nclass SomeType:\n    foobar: strawberry.auto\n\n\nobj1 = SomeType(foobar=1)\nobj2 = SomeType(foobar="some text")\nobj3 = SomeType(foobar={"some key": "some value"})\n\nreveal_type(obj1.foobar)\nreveal_type(obj2.foobar)\nreveal_type(obj3.foobar)\n'

def test_pyright():
    if False:
        while True:
            i = 10
    results = run_pyright(CODE)
    assert results == [Result(type='information', message='Type of "obj1.foobar" is "Any"', line=14, column=13), Result(type='information', message='Type of "obj2.foobar" is "Any"', line=15, column=13), Result(type='information', message='Type of "obj3.foobar" is "Any"', line=16, column=13)]