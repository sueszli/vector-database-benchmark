from pytest import raises
from vyper.exceptions import VyperException

def test_type_exception_pos():
    if False:
        while True:
            i = 10
    pos = (1, 2)
    with raises(VyperException) as e:
        raise VyperException('Fail!', pos)
    assert e.value.lineno == 1
    assert e.value.col_offset == 2
    assert str(e.value) == 'line 1:2 Fail!'

def test_multiple_exceptions(get_contract, assert_compile_failed):
    if False:
        return 10
    code = '\nstruct A:\n    b: B  # unknown type\n\nfoo: immutable(uint256)\nbar: immutable(uint256)\n@external\ndef __init__():\n    self.foo = 1  # SyntaxException\n    self.bar = 2  # SyntaxException\n\n    '
    assert_compile_failed(lambda : get_contract(code), VyperException)