import pytest
from autokeras.engine import block as block_module

def test_block_call_raise_inputs_type_error():
    if False:
        for i in range(10):
            print('nop')
    block = block_module.Block()
    with pytest.raises(TypeError) as info:
        block(None)
    assert 'Expect the inputs to block' in str(info.value)