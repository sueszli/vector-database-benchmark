from jupytext.stringparser import StringParser

def test_long_string(text='\'\'\'This is a multiline\ncomment with "quotes", \'single quotes\'\n# and comments\nand line breaks\n\n\nand it ends here\'\'\'\n\n\n1 + 1\n'):
    if False:
        i = 10
        return i + 15
    quoted = []
    sp = StringParser('python')
    for (i, line) in enumerate(text.splitlines()):
        if sp.is_quoted():
            quoted.append(i)
        sp.read_line(line)
    assert quoted == [1, 2, 3, 4, 5, 6]

def test_single_chars(text='\'This is a single line comment\'\'\'\n\'and another one\'\n# and comments\n"and line breaks"\n\n\n"and it ends here\'\'\'"\n\n\n1 + 1\n'):
    if False:
        for i in range(10):
            print('nop')
    sp = StringParser('python')
    for line in text.splitlines():
        assert not sp.is_quoted()
        sp.read_line(line)

def test_long_string_with_four_quotes(text="''''This is a multiline\ncomment that starts with four quotes\n'''\n\n1 + 1\n"):
    if False:
        print('Hello World!')
    quoted = []
    sp = StringParser('python')
    for (i, line) in enumerate(text.splitlines()):
        if sp.is_quoted():
            quoted.append(i)
        sp.read_line(line)
    assert quoted == [1, 2]

def test_long_string_ends_with_four_quotes(text="'''This is a multiline\ncomment that ends with four quotes\n''''\n\n1 + 1\n"):
    if False:
        while True:
            i = 10
    quoted = []
    sp = StringParser('python')
    for (i, line) in enumerate(text.splitlines()):
        if sp.is_quoted():
            quoted.append(i)
        sp.read_line(line)
    assert quoted == [1, 2]