import os

def test(hatch):
    if False:
        for i in range(10):
            print('nop')
    result = hatch(os.environ['PYAPP_COMMAND_NAME'])
    assert result.exit_code == 0, result.output