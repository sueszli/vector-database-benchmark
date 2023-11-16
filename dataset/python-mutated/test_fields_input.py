from .utils import Result, requires_pyright, run_pyright, skip_on_windows
pytestmark = [skip_on_windows, requires_pyright]
CODE = '\nimport strawberry\n\n\n@strawberry.input\nclass User:\n    name: str\n\n\nUser(name="Patrick")\nUser(n="Patrick")\n\nreveal_type(User)\nreveal_type(User.__init__)\n'

def test_pyright():
    if False:
        print('Hello World!')
    results = run_pyright(CODE)
    assert results == [Result(type='error', message='No parameter named "n"', line=11, column=6), Result(type='error', message='Argument missing for parameter "name"', line=11, column=1), Result(type='information', message='Type of "User" is "type[User]"', line=13, column=13), Result(type='information', message='Type of "User.__init__" is "(self: User, *, name: str) -> None"', line=14, column=13)]