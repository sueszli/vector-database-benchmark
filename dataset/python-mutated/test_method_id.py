def test_method_id_test(get_contract_with_gas_estimation):
    if False:
        return 10
    method_id_test = '\n@external\ndef double(x: int128) -> int128:\n    return x * 2\n\n@external\ndef returnten() -> int128:\n    ans: Bytes[32] = raw_call(self, concat(method_id("double(int128)"), convert(5, bytes32)), gas=50000, max_outsize=32)  # noqa: E501\n    return convert(convert(ans, bytes32), int128)\n    '
    c = get_contract_with_gas_estimation(method_id_test)
    assert c.returnten() == 10

def test_method_id_bytes4(get_contract):
    if False:
        while True:
            i = 10
    code = "\n@external\ndef sig() -> bytes4:\n    return method_id('transfer(address,uint256)', output_type=bytes4)\n    "
    c = get_contract(code)
    sig = c.sig()
    assert sig == b'\xa9\x05\x9c\xbb'

def test_method_id_Bytes4(get_contract):
    if False:
        while True:
            i = 10
    code = "\n@external\ndef sig() -> Bytes[4]:\n    return method_id('transfer(address,uint256)', output_type=Bytes[4])\n    "
    c = get_contract(code)
    sig = c.sig()
    assert sig == b'\xa9\x05\x9c\xbb'

def test_method_id_bytes4_default(get_contract):
    if False:
        print('Hello World!')
    code = "\n@external\ndef sig() -> Bytes[4]:\n    return method_id('transfer(address,uint256)')\n    "
    c = get_contract(code)
    sig = c.sig()
    assert sig == b'\xa9\x05\x9c\xbb'

def test_method_id_invalid_space(get_contract, assert_compile_failed):
    if False:
        for i in range(10):
            print('nop')
    code = "\n@external\ndef sig() -> bytes4:\n    return method_id('transfer(address, uint256)', output_type=bytes4)\n    "
    assert_compile_failed(lambda : get_contract(code))

def test_method_id_invalid_type(get_contract, assert_compile_failed):
    if False:
        return 10
    code = "\n@external\ndef sig() -> bytes32:\n    return method_id('transfer(address,uint256)', output_type=bytes32)\n    "
    assert_compile_failed(lambda : get_contract(code))