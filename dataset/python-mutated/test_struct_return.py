import pytest

def test_nested_struct(get_contract):
    if False:
        print('Hello World!')
    code = '\nstruct Animal:\n    location: address\n    fur: String[32]\n\nstruct Human:\n    location: address\n    animal: Animal\n\n@external\ndef modify_nested_struct(_human: Human) -> Human:\n    human: Human = _human\n\n    # do stuff, edit the structs\n    # (13 is the length of the result)\n    human.animal.fur = slice(concat(human.animal.fur, " is great"), 0, 13)\n\n    return human\n    '
    c = get_contract(code)
    addr1 = '0x1234567890123456789012345678901234567890'
    addr2 = '0x1234567890123456789012345678900000000000'
    assert c.modify_nested_struct({'location': addr1, 'animal': {'location': addr2, 'fur': 'wool'}}) == (addr1, (addr2, 'wool is great'))

def test_nested_single_struct(get_contract):
    if False:
        while True:
            i = 10
    code = '\nstruct Animal:\n    fur: String[32]\n\nstruct Human:\n    animal: Animal\n\n@external\ndef modify_nested_single_struct(_human: Human) -> Human:\n    human: Human = _human\n\n    # do stuff, edit the structs\n    # (13 is the length of the result)\n    human.animal.fur = slice(concat(human.animal.fur, " is great"), 0, 13)\n\n    return human\n    '
    c = get_contract(code)
    assert c.modify_nested_single_struct({'animal': {'fur': 'wool'}}) == (('wool is great',),)

@pytest.mark.parametrize('string', ['a', 'abc', 'abcde', 'potato'])
def test_string_inside_struct(get_contract, string):
    if False:
        for i in range(10):
            print('nop')
    code = f'\nstruct Person:\n    name: String[6]\n    age: uint256\n\n@external\ndef test_return() -> Person:\n    return Person({{ name:"{string}", age:42 }})\n    '
    c1 = get_contract(code)
    code = '\nstruct Person:\n    name: String[6]\n    age: uint256\n\ninterface jsonabi:\n    def test_return() -> Person: view\n\n@external\ndef test_values(a: address) -> Person:\n    return jsonabi(a).test_return()\n    '
    c2 = get_contract(code)
    assert c2.test_values(c1.address) == (string, 42)

@pytest.mark.parametrize('string', ['a', 'abc', 'abcde', 'potato'])
def test_string_inside_single_struct(get_contract, string):
    if False:
        print('Hello World!')
    code = f'\nstruct Person:\n    name: String[6]\n\n@external\ndef test_return() -> Person:\n    return Person({{ name:"{string}"}})\n    '
    c1 = get_contract(code)
    code = '\nstruct Person:\n    name: String[6]\n\ninterface jsonabi:\n    def test_return() -> Person: view\n\n@external\ndef test_values(a: address) -> Person:\n    return jsonabi(a).test_return()\n    '
    c2 = get_contract(code)
    assert c2.test_values(c1.address) == (string,)