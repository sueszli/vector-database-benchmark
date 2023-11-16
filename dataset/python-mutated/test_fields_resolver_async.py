from .utils import Result, requires_pyright, run_pyright, skip_on_windows
pytestmark = [skip_on_windows, requires_pyright]
CODE = '\nimport strawberry\n\nasync def get_user_age() -> int:\n    return 0\n\n\n@strawberry.type\nclass User:\n    name: str\n    age: int = strawberry.field(resolver=get_user_age)\n\n\nUser(name="Patrick")\nUser(n="Patrick")\n\nreveal_type(User)\nreveal_type(User.__init__)\n'

def test_pyright():
    if False:
        for i in range(10):
            print('nop')
    results = run_pyright(CODE)
    assert results == [Result(type='error', message='No parameter named "n"', line=15, column=6), Result(type='error', message='Argument missing for parameter "name"', line=15, column=1), Result(type='information', message='Type of "User" is "type[User]"', line=17, column=13), Result(type='information', message='Type of "User.__init__" is "(self: User, *, name: str) -> None"', line=18, column=13)]