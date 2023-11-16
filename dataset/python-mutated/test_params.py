from .utils import Result, requires_pyright, run_pyright, skip_on_windows
pytestmark = [skip_on_windows, requires_pyright]
CODE = '\nimport strawberry\n\n\n@strawberry.type(name="User")\nclass UserModel:\n    name: str\n\n\n@strawberry.input(name="User")\nclass UserInput:\n    name: str\n\n\nUserModel(name="Patrick")\nUserModel(n="Patrick")\n\nUserInput(name="Patrick")\nUserInput(n="Patrick")\n'

def test_pyright():
    if False:
        i = 10
        return i + 15
    results = run_pyright(CODE)
    assert results == [Result(type='error', message='No parameter named "n"', line=16, column=11), Result(type='error', message='Argument missing for parameter "name"', line=16, column=1), Result(type='error', message='No parameter named "n"', line=19, column=11), Result(type='error', message='Argument missing for parameter "name"', line=19, column=1)]