def test_string_literal_return(get_contract_with_gas_estimation):
    if False:
        return 10
    code = '\n@external\ndef test() -> String[100]:\n    return "hello world!"\n\n\n@external\ndef testb() -> Bytes[100]:\n    return b"hello world!"\n    '
    c = get_contract_with_gas_estimation(code)
    assert c.test() == 'hello world!'
    assert c.testb() == b'hello world!'

def test_string_convert(get_contract_with_gas_estimation):
    if False:
        return 10
    code = '\n@external\ndef testb() -> String[100]:\n    return convert(b"hello world!", String[100])\n\n@external\ndef testbb() -> String[100]:\n    return convert(convert("hello world!", Bytes[100]), String[100])\n    '
    c = get_contract_with_gas_estimation(code)
    assert c.testb() == 'hello world!'
    assert c.testbb() == 'hello world!'

def test_str_assign(get_contract_with_gas_estimation):
    if False:
        for i in range(10):
            print('nop')
    code = '\n@external\ndef test() -> String[100]:\n    a: String[100] = "baba black sheep"\n    return a\n    '
    c = get_contract_with_gas_estimation(code)
    assert c.test() == 'baba black sheep'