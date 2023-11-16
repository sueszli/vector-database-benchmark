import pytest
from vyper.compiler import compile_code
from vyper.exceptions import MemoryAllocationException

def test_memory_overflow():
    if False:
        while True:
            i = 10
    code = '\n@external\ndef zzz(x: DynArray[uint256, 2**59]):  # 2**64 / 32 bytes per word == 2**59\n    y: uint256[7] = [0,0,0,0,0,0,0]\n\n    y[6] = y[5]\n    '
    with pytest.raises(MemoryAllocationException):
        compile_code(code)