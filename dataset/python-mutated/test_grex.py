import inspect
import pytest
import re
from grex import RegExpBuilder

@pytest.mark.parametrize('test_cases,expected_pattern', [pytest.param(['abc', 'abd', 'abe'], '^ab[c-e]$')])
def test_default_settings(test_cases, expected_pattern):
    if False:
        print('Hello World!')
    pattern = RegExpBuilder.from_test_cases(test_cases).build()
    assert pattern == expected_pattern
    for test_case in test_cases:
        assert re.match(pattern, test_case)

@pytest.mark.parametrize('test_cases,expected_pattern', [pytest.param(['My ♥ and 💩 is yours.'], '^My \\u2665 and \\U0001f4a9 is yours\\.$')])
def test_escaping(test_cases, expected_pattern):
    if False:
        return 10
    pattern = RegExpBuilder.from_test_cases(test_cases).with_escaping_of_non_ascii_chars(use_surrogate_pairs=False).build()
    assert pattern == expected_pattern
    for test_case in test_cases:
        assert re.match(pattern, test_case)

@pytest.mark.parametrize('test_cases,expected_pattern', [pytest.param(['My ♥ and 💩 is yours.'], '^My \\u2665 and \\ud83d\\udca9 is yours\\.$')])
def test_escaping_with_surrogate_pairs(test_cases, expected_pattern):
    if False:
        return 10
    pattern = RegExpBuilder.from_test_cases(test_cases).with_escaping_of_non_ascii_chars(use_surrogate_pairs=True).build()
    assert pattern == expected_pattern

@pytest.mark.parametrize('test_cases,expected_pattern', [pytest.param(['efgh', 'abcxy', 'abcw'], '^(abc(xy|w)|efgh)$')])
def test_capturing_groups(test_cases, expected_pattern):
    if False:
        i = 10
        return i + 15
    pattern = RegExpBuilder.from_test_cases(test_cases).with_capturing_groups().build()
    assert pattern == expected_pattern
    for test_case in test_cases:
        assert re.match(pattern, test_case)

@pytest.mark.parametrize('test_cases,expected_pattern', [pytest.param(['efgh', 'abcxy', 'abcw'], '(?:abc(?:xy|w)|efgh)')])
def test_without_anchors(test_cases, expected_pattern):
    if False:
        for i in range(10):
            print('nop')
    pattern = RegExpBuilder.from_test_cases(test_cases).without_anchors().build()
    assert pattern == expected_pattern
    for test_case in test_cases:
        assert re.match(pattern, test_case)

@pytest.mark.parametrize('test_cases,expected_pattern', [pytest.param(['ABC', 'zBC', 'abc', 'AbC', 'aBc'], '(?i)^[az]bc$')])
def test_case_insensitive_matching(test_cases, expected_pattern):
    if False:
        print('Hello World!')
    pattern = RegExpBuilder.from_test_cases(test_cases).with_case_insensitive_matching().build()
    assert pattern == expected_pattern
    for test_case in test_cases:
        assert re.match(pattern, test_case)

@pytest.mark.parametrize('test_cases,expected_pattern', [pytest.param(['[a-z]', '(d,e,f)'], inspect.cleandoc('\n                (?x)\n                ^\n                  (?:\n                    \\(d,e,f\\)\n                    |\n                    \\[a\\-z\\]\n                  )\n                $\n                '))])
def test_verbose_mode(test_cases, expected_pattern):
    if False:
        return 10
    pattern = RegExpBuilder.from_test_cases(test_cases).with_verbose_mode().build()
    assert pattern == expected_pattern
    for test_case in test_cases:
        assert re.match(pattern, test_case)

@pytest.mark.parametrize('test_cases,expected_pattern', [pytest.param(['Ä@Ö€Ü', 'ä@ö€ü', 'Ä@ö€Ü', 'ä@Ö€ü'], inspect.cleandoc('\n                (?ix)\n                ^\n                  ä@ö€ü\n                $\n                '))])
def test_case_insensitive_matching_and_verbose_mode(test_cases, expected_pattern):
    if False:
        while True:
            i = 10
    pattern = RegExpBuilder.from_test_cases(test_cases).with_case_insensitive_matching().with_verbose_mode().build()
    assert pattern == expected_pattern
    for test_case in test_cases:
        assert re.match(pattern, test_case)

@pytest.mark.parametrize('test_cases,expected_pattern', [pytest.param(['a', 'b\nx\nx', 'c'], '^(?:b(?:\\nx){2}|[ac])$')])
def test_conversion_of_repetitions(test_cases, expected_pattern):
    if False:
        print('Hello World!')
    pattern = RegExpBuilder.from_test_cases(test_cases).with_conversion_of_repetitions().build()
    assert pattern == expected_pattern
    for test_case in test_cases:
        assert re.match(pattern, test_case)

@pytest.mark.parametrize('test_cases,expected_pattern', [pytest.param(['My ♥♥♥ and 💩💩 is yours.'], '^My \\u2665{3} and \\U0001f4a9{2} is yours\\.$')])
def test_escaping_and_conversion_of_repetitions(test_cases, expected_pattern):
    if False:
        print('Hello World!')
    pattern = RegExpBuilder.from_test_cases(test_cases).with_escaping_of_non_ascii_chars(use_surrogate_pairs=False).with_conversion_of_repetitions().build()
    assert pattern == expected_pattern
    for test_case in test_cases:
        assert re.match(pattern, test_case)

@pytest.mark.parametrize('test_cases,expected_pattern', [pytest.param(['a1b2c3'], '^a\\db\\dc\\d$')])
def test_conversion_of_digits(test_cases, expected_pattern):
    if False:
        return 10
    pattern = RegExpBuilder.from_test_cases(test_cases).with_conversion_of_digits().build()
    assert pattern == expected_pattern
    for test_case in test_cases:
        assert re.match(pattern, test_case)

@pytest.mark.parametrize('test_cases,expected_pattern', [pytest.param(['a1b2c3'], '^\\D1\\D2\\D3$')])
def test_conversion_of_non_digits(test_cases, expected_pattern):
    if False:
        i = 10
        return i + 15
    pattern = RegExpBuilder.from_test_cases(test_cases).with_conversion_of_non_digits().build()
    assert pattern == expected_pattern
    for test_case in test_cases:
        assert re.match(pattern, test_case)

@pytest.mark.parametrize('test_cases,expected_pattern', [pytest.param(['\n\t', '\r'], '^\\s(?:\\s)?$')])
def test_conversion_of_whitespace(test_cases, expected_pattern):
    if False:
        i = 10
        return i + 15
    pattern = RegExpBuilder.from_test_cases(test_cases).with_conversion_of_whitespace().build()
    assert pattern == expected_pattern
    for test_case in test_cases:
        assert re.match(pattern, test_case)

@pytest.mark.parametrize('test_cases,expected_pattern', [pytest.param(['a1 b2 c3'], '^\\S\\S \\S\\S \\S\\S$')])
def test_conversion_of_non_whitespace(test_cases, expected_pattern):
    if False:
        while True:
            i = 10
    pattern = RegExpBuilder.from_test_cases(test_cases).with_conversion_of_non_whitespace().build()
    assert pattern == expected_pattern
    for test_case in test_cases:
        assert re.match(pattern, test_case)

@pytest.mark.parametrize('test_cases,expected_pattern', [pytest.param(['abc', '1234'], '^\\w\\w\\w(?:\\w)?$')])
def test_conversion_of_words(test_cases, expected_pattern):
    if False:
        i = 10
        return i + 15
    pattern = RegExpBuilder.from_test_cases(test_cases).with_conversion_of_words().build()
    assert pattern == expected_pattern
    for test_case in test_cases:
        assert re.match(pattern, test_case)

@pytest.mark.parametrize('test_cases,expected_pattern', [pytest.param(['abc 1234'], '^abc\\W1234$')])
def test_conversion_of_non_words(test_cases, expected_pattern):
    if False:
        while True:
            i = 10
    pattern = RegExpBuilder.from_test_cases(test_cases).with_conversion_of_non_words().build()
    assert pattern == expected_pattern
    for test_case in test_cases:
        assert re.match(pattern, test_case)

@pytest.mark.parametrize('test_cases,expected_pattern', [pytest.param(['aababab'], '^aababab$'), pytest.param(['aabababab'], '^a(?:ab){4}$')])
def test_minimum_repetitions(test_cases, expected_pattern):
    if False:
        i = 10
        return i + 15
    pattern = RegExpBuilder.from_test_cases(test_cases).with_conversion_of_repetitions().with_minimum_repetitions(3).build()
    assert pattern == expected_pattern
    for test_case in test_cases:
        assert re.match(pattern, test_case)

@pytest.mark.parametrize('test_cases,expected_pattern', [pytest.param(['ababab'], '^ababab$'), pytest.param(['abcabcabc'], '^(?:abc){3}$')])
def test_minimum_substring_length(test_cases, expected_pattern):
    if False:
        print('Hello World!')
    pattern = RegExpBuilder.from_test_cases(test_cases).with_conversion_of_repetitions().with_minimum_substring_length(3).build()
    assert pattern == expected_pattern
    for test_case in test_cases:
        assert re.match(pattern, test_case)

def test_error_for_empty_test_cases():
    if False:
        i = 10
        return i + 15
    with pytest.raises(ValueError) as exception_info:
        RegExpBuilder.from_test_cases([])
    assert exception_info.value.args[0] == 'No test cases have been provided for regular expression generation'

def test_error_for_invalid_minimum_repetitions():
    if False:
        while True:
            i = 10
    with pytest.raises(ValueError) as exception_info:
        RegExpBuilder.from_test_cases(['abcd']).with_minimum_repetitions(-4)
    assert exception_info.value.args[0] == 'Quantity of minimum repetitions must be greater than zero'

def test_error_for_invalid_minimum_substring_length():
    if False:
        return 10
    with pytest.raises(ValueError) as exception_info:
        RegExpBuilder.from_test_cases(['abcd']).with_minimum_substring_length(-2)
    assert exception_info.value.args[0] == 'Minimum substring length must be greater than zero'