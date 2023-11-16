def test_string_map_keys(get_contract):
    if False:
        i = 10
        return i + 15
    code = '\nf:HashMap[String[1], bool]\n@external\ndef test() -> bool:\n    a:String[1] = "a"\n    b:String[1] = "b"\n    self.f[a] = True\n    return self.f[b]  # should return False\n    '
    c = get_contract(code)
    c.test()
    assert c.test() is False

def test_string_map_keys_literals(get_contract):
    if False:
        for i in range(10):
            print('nop')
    code = '\nf:HashMap[String[1], bool]\n@external\ndef test() -> bool:\n    self.f["a"] = True\n    return self.f["b"]  # should return False\n    '
    c = get_contract(code)
    assert c.test() is False