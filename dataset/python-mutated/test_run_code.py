import pytest
from e2b import run_code
from e2b.sandbox.exception import UnsupportedRuntimeException

def test_run_code():
    if False:
        for i in range(10):
            print('nop')
    code = "console.log('hello\\n'.repeat(10)); throw new Error('error')"
    (stdout, stderr) = run_code('Node16', code)
    assert len(stdout) == 60
    assert 'Error: error' in stderr

def test_unsupported_runtime():
    if False:
        print('Hello World!')
    code = "console.log('hello'); throw new Error('error')"
    with pytest.raises(UnsupportedRuntimeException) as e:
        run_code('unsupported', code)