from source_dv_360.fields import sanitize

def test_sanitize_with_pct():
    if False:
        return 10
    string = '% tesT string:'
    sanitized_string = sanitize(string)
    expected_result = 'pct_test_string'
    assert sanitized_string == expected_result

def test_sanitize_trailing_space():
    if False:
        while True:
            i = 10
    string = '% tesT string:    '
    sanitized_string = sanitize(string)
    expected_result = 'pct_test_string'
    assert sanitized_string == expected_result

def test_sanitize_leading_space():
    if False:
        return 10
    string = '  % tesT string:'
    sanitized_string = sanitize(string)
    expected_result = 'pct_test_string'
    assert sanitized_string == expected_result

def test_sanitize_punctuation():
    if False:
        print('Hello World!')
    string = '% tesT string:,;()#$'
    sanitized_string = sanitize(string)
    expected_result = 'pct_test_string'
    assert sanitized_string == expected_result

def test_sanitize_slash():
    if False:
        print('Hello World!')
    string = '% tesT string:/test'
    sanitized_string = sanitize(string)
    expected_result = 'pct_test_string_test'
    assert sanitized_string == expected_result

def test_sanitize_and():
    if False:
        print('Hello World!')
    string = '% tesT string & test'
    sanitized_string = sanitize(string)
    expected_result = 'pct_test_string_and_test'
    assert sanitized_string == expected_result