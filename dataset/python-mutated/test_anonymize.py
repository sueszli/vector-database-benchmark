from ..anonymize import obfuscate_address, obfuscate_email, obfuscate_string

def test_obfuscate_email():
    if False:
        print('Hello World!')
    email = 'abc@gmail.com'
    result = obfuscate_email(email)
    assert result == 'a...@gmail.com'

def test_obfuscate_email_example_email():
    if False:
        return 10
    email = 'abc@example.com'
    result = obfuscate_email(email)
    assert result == 'a...@example.com'

def test_obfuscate_email_no_at_in_email():
    if False:
        while True:
            i = 10
    email = 'abcgmail.com'
    result = obfuscate_email(email)
    assert result == 'a...........'

def test_obfuscate_string():
    if False:
        while True:
            i = 10
    value = 'AbcDef'
    result = obfuscate_string(value)
    assert result == 'A.....'

def test_obfuscate_string_empty_string():
    if False:
        i = 10
        return i + 15
    value = ''
    result = obfuscate_string(value)
    assert result == value

def test_obfuscate_string_phone_string():
    if False:
        print('Hello World!')
    value = '+40123123123'
    result = obfuscate_string(value, phone=True)
    assert result == '+40.........'

def test_obfuscate_address(address):
    if False:
        while True:
            i = 10
    first_name = address.first_name
    last_name = address.last_name
    company_name = address.company_name
    street_address_1 = address.street_address_1
    phone = str(address.phone)
    result = obfuscate_address(address)
    assert result.first_name == first_name[0] + '.' * (len(first_name) - 1)
    assert result.last_name == last_name[0] + '.' * (len(last_name) - 1)
    assert result.company_name == company_name[0] + '.' * (len(company_name) - 1)
    assert result.street_address_1 == street_address_1[0] + '.' * (len(street_address_1) - 1)
    assert result.street_address_2 == ''
    assert result.phone == phone[:3] + '.' * (len(phone) - 3)

def test_obfuscate_address_no_address(address):
    if False:
        i = 10
        return i + 15
    result = obfuscate_address(None)
    assert result is None