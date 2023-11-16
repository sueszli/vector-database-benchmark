from scripts.no_bool_in_generic import check_for_bool_in_generic
BAD_FILE = 'def foo(a: bool) -> bool:\n    return bool(0)\n'
GOOD_FILE = 'def foo(a: bool_t) -> bool_t:\n    return bool(0)\n'

def test_bad_file_with_replace():
    if False:
        return 10
    content = BAD_FILE
    (mutated, result) = check_for_bool_in_generic(content)
    expected = GOOD_FILE
    assert result == expected
    assert mutated

def test_good_file_with_replace():
    if False:
        for i in range(10):
            print('nop')
    content = GOOD_FILE
    (mutated, result) = check_for_bool_in_generic(content)
    expected = content
    assert result == expected
    assert not mutated