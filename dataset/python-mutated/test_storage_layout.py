from vyper.compiler import compile_code

def test_storage_layout():
    if False:
        return 10
    code = '\nfoo: HashMap[address, uint256]\n\n@external\n@nonreentrant("foo")\ndef public_foo1():\n    pass\n\n@external\n@nonreentrant("foo")\ndef public_foo2():\n    pass\n\n\n@internal\n@nonreentrant("bar")\ndef _bar():\n    pass\n\narr: DynArray[uint256, 3]\n\n# mix it up a little\nbaz: Bytes[65]\nbar: uint256\n\n@external\n@nonreentrant("bar")\ndef public_bar():\n    pass\n\n@external\n@nonreentrant("foo")\ndef public_foo3():\n    pass\n    '
    out = compile_code(code, output_formats=['layout'])
    assert out['layout']['storage_layout'] == {'nonreentrant.foo': {'type': 'nonreentrant lock', 'slot': 0}, 'nonreentrant.bar': {'type': 'nonreentrant lock', 'slot': 1}, 'foo': {'type': 'HashMap[address, uint256]', 'slot': 2}, 'arr': {'type': 'DynArray[uint256, 3]', 'slot': 3}, 'baz': {'type': 'Bytes[65]', 'slot': 7}, 'bar': {'type': 'uint256', 'slot': 11}}

def test_storage_and_immutables_layout():
    if False:
        while True:
            i = 10
    code = '\nname: String[32]\nSYMBOL: immutable(String[32])\nDECIMALS: immutable(uint8)\n\n@external\ndef __init__():\n    SYMBOL = "VYPR"\n    DECIMALS = 18\n    '
    expected_layout = {'code_layout': {'DECIMALS': {'length': 32, 'offset': 64, 'type': 'uint8'}, 'SYMBOL': {'length': 64, 'offset': 0, 'type': 'String[32]'}}, 'storage_layout': {'name': {'slot': 0, 'type': 'String[32]'}}}
    out = compile_code(code, output_formats=['layout'])
    assert out['layout'] == expected_layout