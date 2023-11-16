from .utils import Result, requires_pyright, run_pyright, skip_on_windows
pytestmark = [skip_on_windows, requires_pyright]
CODE = '\nimport strawberry\nfrom strawberry.type import StrawberryOptional, StrawberryList\n\n\n@strawberry.type\nclass Fruit:\n    name: str\n\n\nreveal_type(StrawberryOptional(Fruit))\nreveal_type(StrawberryList(Fruit))\nreveal_type(StrawberryOptional(StrawberryList(Fruit)))\nreveal_type(StrawberryList(StrawberryOptional(Fruit)))\n\nreveal_type(StrawberryOptional(str))\nreveal_type(StrawberryList(str))\nreveal_type(StrawberryOptional(StrawberryList(str)))\nreveal_type(StrawberryList(StrawberryOptional(str)))\n'

def test_pyright():
    if False:
        for i in range(10):
            print('nop')
    results = run_pyright(CODE)
    assert results == [Result(type='information', message='Type of "StrawberryOptional(Fruit)" is "StrawberryOptional"', line=11, column=13), Result(type='information', message='Type of "StrawberryList(Fruit)" is "StrawberryList"', line=12, column=13), Result(type='information', message='Type of "StrawberryOptional(StrawberryList(Fruit))" is "StrawberryOptional"', line=13, column=13), Result(type='information', message='Type of "StrawberryList(StrawberryOptional(Fruit))" is "StrawberryList"', line=14, column=13), Result(type='information', message='Type of "StrawberryOptional(str)" is "StrawberryOptional"', line=16, column=13), Result(type='information', message='Type of "StrawberryList(str)" is "StrawberryList"', line=17, column=13), Result(type='information', message='Type of "StrawberryOptional(StrawberryList(str))" is "StrawberryOptional"', line=18, column=13), Result(type='information', message='Type of "StrawberryList(StrawberryOptional(str))" is "StrawberryList"', line=19, column=13)]