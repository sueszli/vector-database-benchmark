from textwrap import dedent

def test_fstring_multiline(Script):
    if False:
        for i in range(10):
            print('nop')
    code = dedent("        '' f'''s{\n           str.uppe\n        '''\n        ")
    (c,) = Script(code).complete(line=2, column=9)
    assert c.name == 'upper'