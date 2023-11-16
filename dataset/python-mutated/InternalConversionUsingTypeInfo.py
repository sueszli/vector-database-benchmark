import unicodedata
from robot.api import TypeInfo

def internal_conversion(type_hint, value, expected):
    if False:
        for i in range(10):
            print('nop')
    assert TypeInfo.from_type_hint(type_hint).convert(value) == expected

def custom_converters(name, expected):
    if False:
        i = 10
        return i + 15

    class Name:
        pass
    info = TypeInfo.from_type_hint(Name)
    converters = {Name: unicodedata.lookup}
    assert info.convert(name, custom_converters=converters) == expected

def language_configuration():
    if False:
        for i in range(10):
            print('nop')
    info = TypeInfo.from_type_hint(bool)
    assert info.convert('kyllä', languages='Finnish') is True
    assert info.convert('ei', languages=['de', 'fi']) is False

def default_language_configuration():
    if False:
        print('Hello World!')
    info = TypeInfo.from_type_hint(bool)
    assert info.convert('ja') is True
    assert info.convert('nein') is False
    assert info.convert('ja', languages='fi') == 'ja'
    assert info.convert('nein', languages='en') == 'nein'