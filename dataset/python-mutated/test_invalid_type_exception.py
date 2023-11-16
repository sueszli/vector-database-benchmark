import pytest
from vyper.exceptions import InvalidType, UnknownType
fail_list = ['\nx: bat\n    ', '\nx: HashMap[int, int128]\n    ', '\nstruct A:\n    b: B\n    ']

@pytest.mark.parametrize('bad_code', fail_list)
def test_unknown_type_exception(bad_code, get_contract, assert_compile_failed):
    if False:
        for i in range(10):
            print('nop')
    assert_compile_failed(lambda : get_contract(bad_code), UnknownType)
invalid_list = ['\n@external\ndef foo():\n    raw_log(b"cow", b"dog")\n    ', '\n@external\ndef foo():\n    xs: uint256[1] = []\n    ', '\n@external\ndef mint(_to: address, _value: uint256):\n    assert msg.sender == self,msg.sender\n    ', '\nevent Foo:\n    message: String[1]\n@external\ndef foo():\n    log Foo("abcd")\n    ', '\n@external\ndef mint(_to: address, _value: uint256):\n    raise 1\n    ', '\nx: int128[3.5]\n    ', '\nb: HashMap[(int128, decimal), int128]\n    ', '\na: constant(address) = 0x3cd751e6b0078be393132286c442345e5dc49699\n    ', '\nx: String <= 33\n    ', '\nx: Bytes <= wei\n    ', '\nx: 5\n    ']

@pytest.mark.parametrize('bad_code', invalid_list)
def test_invalid_type_exception(bad_code, get_contract, assert_compile_failed):
    if False:
        print('Hello World!')
    assert_compile_failed(lambda : get_contract(bad_code), InvalidType)