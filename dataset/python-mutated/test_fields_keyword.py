from .utils import Result, requires_pyright, run_pyright, skip_on_windows
pytestmark = [skip_on_windows, requires_pyright]
CODE = '\nimport strawberry\n\n\n@strawberry.type\nclass User:\n    name: str\n\n\nUser("Patrick")\n\nreveal_type(User.__init__)\n'

def test_pyright():
    if False:
        for i in range(10):
            print('nop')
    results = run_pyright(CODE)
    assert results == [Result(type='error', message='Expected 0 positional arguments', line=10, column=6), Result(type='information', message='Type of "User.__init__" is "(self: User, *, name: str) -> None"', line=12, column=13)]