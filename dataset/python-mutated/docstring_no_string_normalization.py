class ALonelyClass:
    """
    A multiline class docstring.
    """

    def AnEquallyLonelyMethod(self):
        if False:
            while True:
                i = 10
        '\n        A multiline method docstring'
        pass

def one_function():
    if False:
        print('Hello World!')
    'This is a docstring with a single line of text.'
    pass

def shockingly_the_quotes_are_normalized():
    if False:
        return 10
    'This is a multiline docstring.\n    This is a multiline docstring.\n    This is a multiline docstring.\n    '
    pass

def foo():
    if False:
        while True:
            i = 10
    'This is a docstring with             \n  some lines of text here\n  '
    return

def baz():
    if False:
        return 10
    '"This" is a string with some\n  embedded "quotes"'
    return

def poit():
    if False:
        while True:
            i = 10
    '\n  Lorem ipsum dolor sit amet.       \n\n  Consectetur adipiscing elit:\n   - sed do eiusmod tempor incididunt ut labore\n   - dolore magna aliqua\n     - enim ad minim veniam\n     - quis nostrud exercitation ullamco laboris nisi\n   - aliquip ex ea commodo consequat\n  '
    pass

def under_indent():
    if False:
        while True:
            i = 10
    '\n  These lines are indented in a way that does not\nmake sense.\n  '
    pass

def over_indent():
    if False:
        i = 10
        return i + 15
    '\n  This has a shallow indent\n    - But some lines are deeper\n    - And the closing quote is too deep\n    '
    pass

def single_line():
    if False:
        for i in range(10):
            print('nop')
    'But with a newline after it!\n\n    '
    pass

def this():
    if False:
        for i in range(10):
            print('nop')
    "\n    'hey ho'\n    "

def that():
    if False:
        i = 10
        return i + 15
    ' "hey yah" '

def and_that():
    if False:
        return 10
    '\n  "hey yah" '

def and_this():
    if False:
        for i in range(10):
            print('nop')
    ' \n  "hey yah"'

def believe_it_or_not_this_is_in_the_py_stdlib():
    if False:
        for i in range(10):
            print('nop')
    ' \n"hey yah"'

def shockingly_the_quotes_are_normalized_v2():
    if False:
        print('Hello World!')
    '\n    Docstring Docstring Docstring\n    '
    pass

def backslash_space():
    if False:
        print('Hello World!')
    '\\ '

def multiline_backslash_1():
    if False:
        for i in range(10):
            print('nop')
    '\n  hey\there  \\ '

def multiline_backslash_2():
    if False:
        return 10
    '\n  hey there \\ '

def multiline_backslash_3():
    if False:
        while True:
            i = 10
    '\n  already escaped \\ '

class ALonelyClass:
    """
    A multiline class docstring.
    """

    def AnEquallyLonelyMethod(self):
        if False:
            i = 10
            return i + 15
        '\n        A multiline method docstring'
        pass

def one_function():
    if False:
        while True:
            i = 10
    'This is a docstring with a single line of text.'
    pass

def shockingly_the_quotes_are_normalized():
    if False:
        while True:
            i = 10
    'This is a multiline docstring.\n    This is a multiline docstring.\n    This is a multiline docstring.\n    '
    pass

def foo():
    if False:
        i = 10
        return i + 15
    'This is a docstring with\n    some lines of text here\n    '
    return

def baz():
    if False:
        i = 10
        return i + 15
    '"This" is a string with some\n    embedded "quotes"'
    return

def poit():
    if False:
        while True:
            i = 10
    '\n    Lorem ipsum dolor sit amet.\n\n    Consectetur adipiscing elit:\n     - sed do eiusmod tempor incididunt ut labore\n     - dolore magna aliqua\n       - enim ad minim veniam\n       - quis nostrud exercitation ullamco laboris nisi\n     - aliquip ex ea commodo consequat\n    '
    pass

def under_indent():
    if False:
        for i in range(10):
            print('nop')
    '\n      These lines are indented in a way that does not\n    make sense.\n    '
    pass

def over_indent():
    if False:
        return 10
    '\n    This has a shallow indent\n      - But some lines are deeper\n      - And the closing quote is too deep\n    '
    pass

def single_line():
    if False:
        return 10
    'But with a newline after it!'
    pass

def this():
    if False:
        while True:
            i = 10
    "\n    'hey ho'\n    "

def that():
    if False:
        while True:
            i = 10
    ' "hey yah" '

def and_that():
    if False:
        return 10
    '\n    "hey yah" '

def and_this():
    if False:
        print('Hello World!')
    '\n    "hey yah"'

def believe_it_or_not_this_is_in_the_py_stdlib():
    if False:
        return 10
    '\n    "hey yah"'

def shockingly_the_quotes_are_normalized_v2():
    if False:
        while True:
            i = 10
    '\n    Docstring Docstring Docstring\n    '
    pass

def backslash_space():
    if False:
        for i in range(10):
            print('nop')
    '\\ '

def multiline_backslash_1():
    if False:
        return 10
    '\n  hey\there  \\ '

def multiline_backslash_2():
    if False:
        return 10
    '\n    hey there \\ '

def multiline_backslash_3():
    if False:
        print('Hello World!')
    '\n    already escaped \\'