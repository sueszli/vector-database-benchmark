from .utils import Result, requires_pyright, run_pyright, skip_on_windows
pytestmark = [skip_on_windows, requires_pyright]
CODE = '\nimport strawberry\n\n\n@strawberry.federation.type(name="User")\nclass UserModel:\n    name: str\n\n\nUserModel(name="Patrick")\nUserModel(n="Patrick")\n'

def test_pyright():
    if False:
        for i in range(10):
            print('nop')
    results = run_pyright(CODE)
    assert results == [Result(type='error', message='No parameter named "n"', line=11, column=11), Result(type='error', message='Argument missing for parameter "name"', line=11, column=1)]