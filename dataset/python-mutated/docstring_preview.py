def docstring_almost_at_line_limit():
    if False:
        i = 10
        return i + 15
    'long docstring.................................................................\n    '

def docstring_almost_at_line_limit_with_prefix():
    if False:
        print('Hello World!')
    f'long docstring................................................................\n    '

def mulitline_docstring_almost_at_line_limit():
    if False:
        print('Hello World!')
    'long docstring.................................................................\n\n    ..................................................................................\n    '

def mulitline_docstring_almost_at_line_limit_with_prefix():
    if False:
        for i in range(10):
            print('nop')
    f'long docstring................................................................\n\n    ..................................................................................\n    '

def docstring_at_line_limit():
    if False:
        return 10
    'long docstring................................................................'

def docstring_at_line_limit_with_prefix():
    if False:
        for i in range(10):
            print('nop')
    f'long docstring...............................................................'

def multiline_docstring_at_line_limit():
    if False:
        return 10
    'first line-----------------------------------------------------------------------\n\n    second line----------------------------------------------------------------------'

def multiline_docstring_at_line_limit_with_prefix():
    if False:
        return 10
    f'first line----------------------------------------------------------------------\n\n    second line----------------------------------------------------------------------'

def single_quote_docstring_over_line_limit():
    if False:
        print('Hello World!')
    'We do not want to put the closing quote on a new line as that is invalid (see GH-3141).'

def single_quote_docstring_over_line_limit2():
    if False:
        print('Hello World!')
    'We do not want to put the closing quote on a new line as that is invalid (see GH-3141).'

def docstring_almost_at_line_limit():
    if False:
        i = 10
        return i + 15
    'long docstring.................................................................'

def docstring_almost_at_line_limit_with_prefix():
    if False:
        for i in range(10):
            print('nop')
    f'long docstring................................................................\n    '

def mulitline_docstring_almost_at_line_limit():
    if False:
        i = 10
        return i + 15
    'long docstring.................................................................\n\n    ..................................................................................\n    '

def mulitline_docstring_almost_at_line_limit_with_prefix():
    if False:
        i = 10
        return i + 15
    f'long docstring................................................................\n\n    ..................................................................................\n    '

def docstring_at_line_limit():
    if False:
        return 10
    'long docstring................................................................'

def docstring_at_line_limit_with_prefix():
    if False:
        while True:
            i = 10
    f'long docstring...............................................................'

def multiline_docstring_at_line_limit():
    if False:
        while True:
            i = 10
    'first line-----------------------------------------------------------------------\n\n    second line----------------------------------------------------------------------'

def multiline_docstring_at_line_limit_with_prefix():
    if False:
        while True:
            i = 10
    f'first line----------------------------------------------------------------------\n\n    second line----------------------------------------------------------------------'

def single_quote_docstring_over_line_limit():
    if False:
        i = 10
        return i + 15
    'We do not want to put the closing quote on a new line as that is invalid (see GH-3141).'

def single_quote_docstring_over_line_limit2():
    if False:
        for i in range(10):
            print('nop')
    'We do not want to put the closing quote on a new line as that is invalid (see GH-3141).'