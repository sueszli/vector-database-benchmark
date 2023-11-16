def test():
    if False:
        i = 10
        return i + 15
    foo('a string')
    foo('another string')
    foo('"escaped string"')
    foo(f'an fstring')
    foo('a multiline string')
    foo('singlequote with " inside')
    foo("doublequote with ' inside")
    foo(1)