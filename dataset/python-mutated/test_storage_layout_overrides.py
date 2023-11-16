import pytest
from vyper.compiler import compile_code
from vyper.exceptions import StorageLayoutException

def test_storage_layout_overrides():
    if False:
        return 10
    code = '\na: uint256\nb: uint256'
    storage_layout_overrides = {'a': {'type': 'uint256', 'slot': 1}, 'b': {'type': 'uint256', 'slot': 0}}
    expected_output = {'storage_layout': storage_layout_overrides, 'code_layout': {}}
    out = compile_code(code, output_formats=['layout'], storage_layout_override=storage_layout_overrides)
    assert out['layout'] == expected_output

def test_storage_layout_for_more_complex():
    if False:
        while True:
            i = 10
    code = '\nfoo: HashMap[address, uint256]\n\n@external\n@nonreentrant("foo")\ndef public_foo1():\n    pass\n\n@external\n@nonreentrant("foo")\ndef public_foo2():\n    pass\n\n\n@internal\n@nonreentrant("bar")\ndef _bar():\n    pass\n\n# mix it up a little\nbaz: Bytes[65]\nbar: uint256\n\n@external\n@nonreentrant("bar")\ndef public_bar():\n    pass\n\n@external\n@nonreentrant("foo")\ndef public_foo3():\n    pass\n    '
    storage_layout_override = {'nonreentrant.foo': {'type': 'nonreentrant lock', 'slot': 8}, 'nonreentrant.bar': {'type': 'nonreentrant lock', 'slot': 7}, 'foo': {'type': 'HashMap[address, uint256]', 'slot': 1}, 'baz': {'type': 'Bytes[65]', 'slot': 2}, 'bar': {'type': 'uint256', 'slot': 6}}
    expected_output = {'storage_layout': storage_layout_override, 'code_layout': {}}
    out = compile_code(code, output_formats=['layout'], storage_layout_override=storage_layout_override)
    assert out['layout'] == expected_output

def test_simple_collision():
    if False:
        return 10
    code = '\nname: public(String[64])\nsymbol: public(String[32])'
    storage_layout_override = {'name': {'slot': 0, 'type': 'String[64]'}, 'symbol': {'slot': 1, 'type': 'String[32]'}}
    with pytest.raises(StorageLayoutException, match="Storage collision! Tried to assign 'symbol' to slot 1 but it has already been reserved by 'name'"):
        compile_code(code, output_formats=['layout'], storage_layout_override=storage_layout_override)

def test_overflow():
    if False:
        while True:
            i = 10
    code = '\nx: uint256[2]\n    '
    storage_layout_override = {'x': {'slot': 2 ** 256 - 1, 'type': 'uint256[2]'}}
    with pytest.raises(StorageLayoutException, match=f'Invalid storage slot for var x, out of bounds: {2 ** 256}\n'):
        compile_code(code, output_formats=['layout'], storage_layout_override=storage_layout_override)

def test_incomplete_overrides():
    if False:
        print('Hello World!')
    code = '\nname: public(String[64])\nsymbol: public(String[32])'
    storage_layout_override = {'name': {'slot': 0, 'type': 'String[64]'}}
    with pytest.raises(StorageLayoutException, match='Could not find storage_slot for symbol. Have you used the correct storage layout file?'):
        compile_code(code, output_formats=['layout'], storage_layout_override=storage_layout_override)