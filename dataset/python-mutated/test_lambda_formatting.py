from hypothesis.internal.reflection import get_pretty_function_description

def test_bracket_whitespace_is_striped():
    if False:
        for i in range(10):
            print('nop')
    assert get_pretty_function_description(lambda x: x + 1) == 'lambda x: (x + 1)'

def test_no_whitespace_before_colon_with_no_args():
    if False:
        return 10
    assert get_pretty_function_description(eval('lambda: None')) == 'lambda: <unknown>'

def test_can_have_unicode_in_lambda_sources():
    if False:
        i = 10
        return i + 15
    t = lambda x: 'é' not in x
    assert get_pretty_function_description(t) == 'lambda x: "é" not in x'
ordered_pair = lambda right: [].map(lambda length: ())

def test_can_get_descriptions_of_nested_lambdas_with_different_names():
    if False:
        print('Hello World!')
    assert get_pretty_function_description(ordered_pair) == 'lambda right: [].map(lambda length: ())'

def test_does_not_error_on_unparsable_source():
    if False:
        for i in range(10):
            print('nop')
    t = [lambda x: x][0]
    assert get_pretty_function_description(t) == 'lambda x: <unknown>'

def test_source_of_lambda_is_pretty():
    if False:
        i = 10
        return i + 15
    assert get_pretty_function_description(lambda x: True) == 'lambda x: True'

def test_variable_names_are_not_pretty():
    if False:
        while True:
            i = 10
    t = lambda x: True
    assert get_pretty_function_description(t) == 'lambda x: True'

def test_does_not_error_on_dynamically_defined_functions():
    if False:
        i = 10
        return i + 15
    x = eval('lambda t: 1')
    get_pretty_function_description(x)

def test_collapses_whitespace_nicely():
    if False:
        i = 10
        return i + 15
    t = lambda x, y: 1
    assert get_pretty_function_description(t) == 'lambda x, y: 1'

def test_is_not_confused_by_tuples():
    if False:
        i = 10
        return i + 15
    p = (lambda x: x > 1, 2)[0]
    assert get_pretty_function_description(p) == 'lambda x: x > 1'

def test_strips_comments_from_the_end():
    if False:
        while True:
            i = 10
    t = lambda x: 1
    assert get_pretty_function_description(t) == 'lambda x: 1'

def test_does_not_strip_hashes_within_a_string():
    if False:
        i = 10
        return i + 15
    t = lambda x: '#'
    assert get_pretty_function_description(t) == 'lambda x: "#"'

def test_can_distinguish_between_two_lambdas_with_different_args():
    if False:
        return 10
    (a, b) = (lambda x: 1, lambda y: 2)
    assert get_pretty_function_description(a) == 'lambda x: 1'
    assert get_pretty_function_description(b) == 'lambda y: 2'

def test_does_not_error_if_it_cannot_distinguish_between_two_lambdas():
    if False:
        i = 10
        return i + 15
    (a, b) = (lambda x: 1, lambda x: 2)
    assert 'lambda x:' in get_pretty_function_description(a)
    assert 'lambda x:' in get_pretty_function_description(b)

def test_lambda_source_break_after_def_with_brackets():
    if False:
        while True:
            i = 10
    f = lambda n: 'aaa'
    source = get_pretty_function_description(f)
    assert source == "lambda n: 'aaa'"

def test_lambda_source_break_after_def_with_line_continuation():
    if False:
        i = 10
        return i + 15
    f = lambda n: 'aaa'
    source = get_pretty_function_description(f)
    assert source == "lambda n: 'aaa'"

def arg_decorator(*s):
    if False:
        for i in range(10):
            print('nop')

    def accept(f):
        if False:
            print('Hello World!')
        return s
    return accept

@arg_decorator(lambda x: x + 1)
def plus_one():
    if False:
        print('Hello World!')
    pass

@arg_decorator(lambda x: x + 1, lambda y: y * 2)
def two_decorators():
    if False:
        return 10
    pass

def test_can_extract_lambda_repr_in_a_decorator():
    if False:
        return 10
    assert get_pretty_function_description(plus_one[0]) == 'lambda x: x + 1'

def test_can_extract_two_lambdas_from_a_decorator_if_args_differ():
    if False:
        return 10
    (a, b) = two_decorators
    assert get_pretty_function_description(a) == 'lambda x: x + 1'
    assert get_pretty_function_description(b) == 'lambda y: y * 2'

@arg_decorator(lambda x: x + 1)
def decorator_with_space():
    if False:
        while True:
            i = 10
    pass

def test_can_extract_lambda_repr_in_a_decorator_with_spaces():
    if False:
        while True:
            i = 10
    assert get_pretty_function_description(decorator_with_space[0]) == 'lambda x: x + 1'

@arg_decorator(lambda : ())
def to_brackets():
    if False:
        for i in range(10):
            print('nop')
    pass

def test_can_handle_brackets_in_decorator_argument():
    if False:
        while True:
            i = 10
    assert get_pretty_function_description(to_brackets[0]) == 'lambda: ()'

def identity(x):
    if False:
        print('Hello World!')
    return x

@arg_decorator(identity(lambda x: x + 1))
def decorator_with_wrapper():
    if False:
        print('Hello World!')
    pass

def test_can_handle_nested_lambda_in_decorator_argument():
    if False:
        while True:
            i = 10
    assert get_pretty_function_description(decorator_with_wrapper[0]) == 'lambda x: x + 1'