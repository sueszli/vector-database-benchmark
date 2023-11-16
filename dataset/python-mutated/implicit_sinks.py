from builtins import _test_source

def propagate_sink_format_string(a):
    if False:
        while True:
            i = 10
    f'<{a}>'

def inline_issue_format_string():
    if False:
        return 10
    a = _test_source()
    f'<{a}>'
    f'{a}'

def propagate_sink_dot_format(a):
    if False:
        for i in range(10):
            print('nop')
    '<{}>'.format(a)

def inline_issue_dot_format():
    if False:
        return 10
    a = _test_source()
    '<{}>'.format(a)

def propagate_sink_percent_format(a):
    if False:
        i = 10
        return i + 15
    '<%s>' % (a,)

def inline_issue_percent_format():
    if False:
        print('Hello World!')
    a = _test_source()
    '<%s>' % (a,)

def propagate_sink_rhs_add_literal(a):
    if False:
        return 10
    'https://' + a

def inline_issue_rhs_add_literal():
    if False:
        while True:
            i = 10
    a = _test_source()
    'https://' + a
https_start = 'https://'

def propagate_sink_add_global(a):
    if False:
        print('Hello World!')
    https_start + a

def propagate_sink_lhs_add_literal(a):
    if False:
        return 10
    columns = a + ' FROM'

def inline_issue_lhs_add_literal():
    if False:
        for i in range(10):
            print('nop')
    a = _test_source()
    columns = a + ' FROM'

def inline_issue_format_string_proper_tito():
    if False:
        print('Hello World!')
    (a, b, c) = (_test_source(), '', _test_source())
    f'<{a}{b}{c}>'

def implicit_sink_before_source():
    if False:
        while True:
            i = 10
    a = '<{}>'
    a.format(_test_source())

def implicit_sink_before_parameter(y):
    if False:
        print('Hello World!')
    a = '<{}>'
    a.format(y)

def format_wrapper(a, y):
    if False:
        return 10
    a.format(y)

def conditional_literal_sink():
    if False:
        while True:
            i = 10
    y = _test_source()
    a = '<{}>'
    format_wrapper(a, y)

def string_literal_arguments_sink(template: str):
    if False:
        i = 10
        return i + 15
    x = _test_source()
    if 1 == 1:
        template.format('https://1', x)
    elif 1 == 1:
        template % ('https://2', x)
    else:
        x + 'https://3'

def string_literal_arguments_issue():
    if False:
        return 10
    string_literal_arguments_sink(_test_source())