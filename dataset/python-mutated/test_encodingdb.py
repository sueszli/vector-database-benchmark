"""Tests based on the Adobe Glyph List Specification
See: https://github.com/adobe-type-tools/agl-specification#2-the-mapping

While not in the specification, lowercase unicode often occurs in pdf's.
Therefore lowercase unittest variants are added.
"""
import pytest
from pdfminer.encodingdb import name2unicode, EncodingDB
from pdfminer.psparser import PSLiteral

def test_name2unicode_name_in_agl():
    if False:
        i = 10
        return i + 15
    'The name "Lcommaaccent" has a single component,\n    which is mapped to the string U+013B by AGL'
    assert 'Ļ' == name2unicode('Lcommaaccent')

def test_name2unicode_uni():
    if False:
        while True:
            i = 10
    'The components "Lcommaaccent," "uni013B," and "u013B"\n    all map to the string U+013B'
    assert 'Ļ' == name2unicode('uni013B')

def test_name2unicode_uni_lowercase():
    if False:
        i = 10
        return i + 15
    'The components "Lcommaaccent," "uni013B," and "u013B"\n    all map to the string U+013B'
    assert 'Ļ' == name2unicode('uni013b')

def test_name2unicode_uni_with_sequence_of_digits():
    if False:
        i = 10
        return i + 15
    'The name "uni20AC0308" has a single component,\n    which is mapped to the string U+20AC U+0308'
    assert '€̈' == name2unicode('uni20AC0308')

def test_name2unicode_uni_with_sequence_of_digits_lowercase():
    if False:
        i = 10
        return i + 15
    'The name "uni20AC0308" has a single component,\n    which is mapped to the string U+20AC U+0308'
    assert '€̈' == name2unicode('uni20ac0308')

def test_name2unicode_uni_empty_string():
    if False:
        print('Hello World!')
    'The name "uni20ac" has a single component,\n    which is mapped to a euro-sign.\n\n    According to the specification this should be mapped to an empty string,\n    but we also want to support lowercase hexadecimals'
    assert '€' == name2unicode('uni20ac')

def test_name2unicode_uni_empty_string_long():
    if False:
        i = 10
        return i + 15
    'The name "uniD801DC0C" has a single component,\n    which is mapped to an empty string\n\n    Neither D801 nor DC0C are in the appropriate set.\n    This form cannot be used to map to the character which is\n    expressed as D801 DC0C in UTF-16, specifically U+1040C.\n    This character can be correctly mapped by using the\n    glyph name "u1040C.\n    '
    with pytest.raises(KeyError):
        name2unicode('uniD801DC0C')

def test_name2unicode_uni_empty_string_long_lowercase():
    if False:
        for i in range(10):
            print('nop')
    'The name "uniD801DC0C" has a single component,\n    which is mapped to an empty string\n\n    Neither D801 nor DC0C are in the appropriate set.\n    This form cannot be used to map to the character which is\n    expressed as D801 DC0C in UTF-16, specifically U+1040C.\n    This character can be correctly mapped by using the\n    glyph name "u1040C.'
    with pytest.raises(KeyError):
        name2unicode('uniD801DC0C')

def test_name2unicode_uni_pua():
    if False:
        for i in range(10):
            print('nop')
    ' "Ogoneksmall" and "uniF6FB" both map to the string that corresponds to\n    U+F6FB.'
    assert '\uf6fb' == name2unicode('uniF6FB')

def test_name2unicode_uni_pua_lowercase():
    if False:
        while True:
            i = 10
    ' "Ogoneksmall" and "uniF6FB" both map to the string that corresponds to\n    U+F6FB.'
    assert '\uf6fb' == name2unicode('unif6fb')

def test_name2unicode_u_with_4_digits():
    if False:
        for i in range(10):
            print('nop')
    'The components "Lcommaaccent," "uni013B," and "u013B" all map to the\n    string U+013B'
    assert 'Ļ' == name2unicode('u013B')

def test_name2unicode_u_with_4_digits_lowercase():
    if False:
        while True:
            i = 10
    'The components "Lcommaaccent," "uni013B," and "u013B" all map to the\n    string U+013B'
    assert 'Ļ' == name2unicode('u013b')

def test_name2unicode_u_with_5_digits():
    if False:
        print('Hello World!')
    'The name "u1040C" has a single component, which is mapped to the string\n    U+1040C'
    assert '𐐌' == name2unicode('u1040C')

def test_name2unicode_u_with_5_digits_lowercase():
    if False:
        while True:
            i = 10
    'The name "u1040C" has a single component, which is mapped to the string\n    U+1040C'
    assert '𐐌' == name2unicode('u1040c')

def test_name2unicode_multiple_components():
    if False:
        i = 10
        return i + 15
    'The name "Lcommaaccent_uni20AC0308_u1040C.alternate" is mapped to the\n    string U+013B U+20AC U+0308 U+1040C'
    assert 'Ļ€̈𐐌' == name2unicode('Lcommaaccent_uni20AC0308_u1040C.alternate')

def test_name2unicode_multiple_components_lowercase():
    if False:
        for i in range(10):
            print('nop')
    'The name "Lcommaaccent_uni20AC0308_u1040C.alternate" is mapped to the\n    string U+013B U+20AC U+0308 U+1040C'
    assert 'Ļ€̈𐐌' == name2unicode('Lcommaaccent_uni20ac0308_u1040c.alternate')

def test_name2unicode_foo():
    if False:
        print('Hello World!')
    "The name 'foo' maps to an empty string,\n    because 'foo' is not in AGL,\n    and because it does not start with a 'u.'"
    with pytest.raises(KeyError):
        name2unicode('foo')

def test_name2unicode_notdef():
    if False:
        for i in range(10):
            print('nop')
    'The name ".notdef" is reduced to an empty string (step 1)\n    and mapped to an empty string (step 3)'
    with pytest.raises(KeyError):
        name2unicode('.notdef')

def test_name2unicode_pua_ogoneksmall():
    if False:
        print('Hello World!')
    ' "\n    Ogoneksmall" and "uniF6FB" both map to the string\n    that corresponds to U+F6FB.'
    assert '\uf6fb' == name2unicode('Ogoneksmall')

def test_name2unicode_overflow_error():
    if False:
        while True:
            i = 10
    with pytest.raises(KeyError):
        name2unicode('226215240241240240240240')

def test_get_encoding_with_invalid_differences():
    if False:
        print('Hello World!')
    'Invalid differences should be silently ignored\n\n    Regression test for https://github.com/pdfminer/pdfminer.six/issues/385\n    '
    invalid_differences = [PSLiteral('ubuntu'), PSLiteral('1234')]
    EncodingDB.get_encoding('StandardEncoding', invalid_differences)