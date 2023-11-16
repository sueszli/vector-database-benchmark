import pytest
from vyper import compiler
from vyper.exceptions import StructureException
fail_list = [('\nfoo: DynArray[HashMap[uint8, uint8], 2]\n    ', StructureException), ('\nfoo: public(DynArray[HashMap[uint8, uint8], 2])\n    ', StructureException), ('\n@external\ndef foo():\n    a: DynArray = [1, 2, 3]\n    ', StructureException)]

@pytest.mark.parametrize('bad_code,exc', fail_list)
def test_block_fail(assert_compile_failed, get_contract, bad_code, exc):
    if False:
        i = 10
        return i + 15
    assert_compile_failed(lambda : get_contract(bad_code), exc)
valid_list = ['\nenum Foo:\n    FE\n    FI\n\nbar: DynArray[Foo, 10]\n    ', '\nbar: DynArray[Bytes[30], 10]\n    ']

@pytest.mark.parametrize('good_code', valid_list)
def test_dynarray_pass(good_code):
    if False:
        return 10
    assert compiler.compile_code(good_code) is not None