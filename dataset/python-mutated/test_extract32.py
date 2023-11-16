def test_extract32_extraction(assert_tx_failed, get_contract_with_gas_estimation):
    if False:
        i = 10
        return i + 15
    extract32_code = '\ny: Bytes[100]\n@external\ndef extrakt32(inp: Bytes[100], index: uint256) -> bytes32:\n    return extract32(inp, index)\n\n@external\ndef extrakt32_mem(inp: Bytes[100], index: uint256) -> bytes32:\n    x: Bytes[100] = inp\n    return extract32(x, index)\n\n@external\ndef extrakt32_storage(index: uint256, inp: Bytes[100]) -> bytes32:\n    self.y = inp\n    return extract32(self.y, index)\n    '
    c = get_contract_with_gas_estimation(extract32_code)
    test_cases = ((b'c' * 31, 0), (b'c' * 32, 0), (b'c' * 33, 0), (b'c' * 33, 1), (b'c' * 33, 2), (b'cow' * 30, 0), (b'cow' * 30, 1), (b'cow' * 30, 31), (b'cow' * 30, 32), (b'cow' * 30, 33), (b'cow' * 30, 34), (b'cow' * 30, 58), (b'cow' * 30, 59))
    for (S, i) in test_cases:
        expected_result = S[i:i + 32] if 0 <= i <= len(S) - 32 else None
        if expected_result is None:
            assert_tx_failed(lambda : c.extrakt32(S, i))
        else:
            assert c.extrakt32(S, i) == expected_result
            assert c.extrakt32_mem(S, i) == expected_result
            assert c.extrakt32_storage(i, S) == expected_result
    print('Passed bytes32 extraction test')

def test_extract32_code(assert_tx_failed, get_contract_with_gas_estimation):
    if False:
        print('Hello World!')
    extract32_code = '\n@external\ndef foo(inp: Bytes[32]) -> int128:\n    return extract32(inp, 0, output_type=int128)\n\n@external\ndef bar(inp: Bytes[32]) -> uint256:\n    return extract32(inp, 0, output_type=uint256)\n\n@external\ndef baz(inp: Bytes[32]) -> bytes32:\n    return extract32(inp, 0, output_type=bytes32)\n\n@external\ndef fop(inp: Bytes[32]) -> bytes32:\n    return extract32(inp, 0)\n\n@external\ndef foq(inp: Bytes[32]) -> address:\n    return extract32(inp, 0, output_type=address)\n    '
    c = get_contract_with_gas_estimation(extract32_code)
    assert c.foo(b'\x00' * 30 + b'\x01\x01') == 257
    assert c.bar(b'\x00' * 30 + b'\x01\x01') == 257
    assert_tx_failed(lambda : c.foo(b'\x80' + b'\x00' * 30))
    assert c.bar(b'\x80' + b'\x00' * 31) == 2 ** 255
    assert c.baz(b'crow' * 8) == b'crow' * 8
    assert c.fop(b'crow' * 8) == b'crow' * 8
    assert c.foq(b'\x00' * 12 + b'3' * 20) == '0x' + '3' * 40
    assert_tx_failed(lambda : c.foq(b'crow' * 8))
    print('Passed extract32 test')