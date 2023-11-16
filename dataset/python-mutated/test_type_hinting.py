import mypy.api

def test_mypy_import():
    if False:
        i = 10
        return i + 15
    (_, _, result) = mypy.api.run(['--strict', '-c', 'from loguru import logger'])
    assert result == 0