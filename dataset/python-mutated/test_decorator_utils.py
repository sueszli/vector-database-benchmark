from dagster._core.decorator_utils import format_docstring_for_description, get_function_params, validate_expected_params

def decorated_function_one_positional():
    if False:
        for i in range(10):
            print('nop')

    def foo(bar):
        if False:
            while True:
                i = 10
        return bar
    return foo

def decorated_function_two_positionals_one_kwarg():
    if False:
        for i in range(10):
            print('nop')

    def foo_kwarg(bar, baz, qux=True):
        if False:
            print('Hello World!')
        return (bar, baz, qux)
    return foo_kwarg

def test_one_required_positional_param():
    if False:
        return 10
    positionals = ['bar']
    fn_params = get_function_params(decorated_function_one_positional())
    assert {fn_param.name for fn_param in fn_params} == {'bar'}
    assert not validate_expected_params(fn_params, positionals)

def test_required_positional_parameters_not_missing():
    if False:
        for i in range(10):
            print('nop')
    positionals = ['bar', 'baz']
    fn_params = get_function_params(decorated_function_two_positionals_one_kwarg())
    assert {fn_param.name for fn_param in fn_params} == {'bar', 'qux', 'baz'}
    assert not validate_expected_params(fn_params, positionals)
    fn_params = get_function_params(decorated_function_one_positional())
    assert validate_expected_params(fn_params, positionals) == 'baz'

def test_format_docstring_for_description():
    if False:
        i = 10
        return i + 15

    def multiline_indented_docstring():
        if False:
            for i in range(10):
                print('nop')
        'abc\n        123.\n        '
    multiline_indented_docstring_expected = 'abc\n123.'
    assert format_docstring_for_description(multiline_indented_docstring) == multiline_indented_docstring_expected

    def no_indentation_at_start():
        if False:
            while True:
                i = 10
        'abc\n        123.\n        '
    no_indentation_at_start_expected = 'abc\n123.'
    assert format_docstring_for_description(no_indentation_at_start) == no_indentation_at_start_expected

    def indentation_at_start():
        if False:
            i = 10
            return i + 15
        '\n        abc\n        123.\n        '
    indentation_at_start_expected = 'abc\n123.'
    assert format_docstring_for_description(indentation_at_start) == indentation_at_start_expected

    def summary_line_and_description():
        if False:
            print('Hello World!')
        "This is the summary line.\n\n        This is a longer description of what my asset does, and I'd like for the\n        newline between this paragraph and the summary line to be preserved.\n        "
    indentation_at_start_expected = "This is the summary line.\n\nThis is a longer description of what my asset does, and I'd like for the\nnewline between this paragraph and the summary line to be preserved."
    assert format_docstring_for_description(summary_line_and_description) == indentation_at_start_expected

def test_empty():
    if False:
        for i in range(10):
            print('nop')

    def empty_docstring():
        if False:
            while True:
                i = 10
        ''
    assert format_docstring_for_description(empty_docstring) == ''