import tokenize
from IPython.testing import tools as tt
from IPython.core import inputtransformer as ipt

def transform_and_reset(transformer):
    if False:
        print('Hello World!')
    transformer = transformer()

    def transform(inp):
        if False:
            print('Hello World!')
        try:
            return transformer.push(inp)
        finally:
            transformer.reset()
    return transform

def transform_checker(tests, transformer, **kwargs):
    if False:
        i = 10
        return i + 15
    'Utility to loop over test inputs'
    transformer = transformer(**kwargs)
    try:
        for (inp, tr) in tests:
            if inp is None:
                out = transformer.reset()
            else:
                out = transformer.push(inp)
            assert out == tr
    finally:
        transformer.reset()
syntax = dict(assign_system=[('a =! ls', "a = get_ipython().getoutput('ls')"), ('b = !ls', "b = get_ipython().getoutput('ls')"), ('c= !ls', "c = get_ipython().getoutput('ls')"), ('d == !ls', 'd == !ls'), ('x=1', 'x=1'), ('    ', '    '), ("a, b = !echo 'a\\nb'", 'a, b = get_ipython().getoutput("echo \'a\\\\nb\'")'), ("a,= !echo 'a'", 'a, = get_ipython().getoutput("echo \'a\'")'), ("a, *bc = !echo 'a\\nb\\nc'", 'a, *bc = get_ipython().getoutput("echo \'a\\\\nb\\\\nc\'")'), ('a, b = range(2)', 'a, b = range(2)'), ('a, = range(1)', 'a, = range(1)'), ('a, *bc = range(3)', 'a, *bc = range(3)')], assign_magic=[('a =% who', "a = get_ipython().run_line_magic('who', '')"), ('b = %who', "b = get_ipython().run_line_magic('who', '')"), ('c= %ls', "c = get_ipython().run_line_magic('ls', '')"), ('d == %ls', 'd == %ls'), ('x=1', 'x=1'), ('    ', '    '), ('a, b = %foo', "a, b = get_ipython().run_line_magic('foo', '')")], classic_prompt=[('>>> x=1', 'x=1'), ('x=1', 'x=1'), ('    ', '    ')], ipy_prompt=[('In [1]: x=1', 'x=1'), ('x=1', 'x=1'), ('    ', '    ')], escaped_noesc=[('    ', '    '), ('x=1', 'x=1')], escaped_shell=[('!ls', "get_ipython().system('ls')"), ('!!ls', "get_ipython().getoutput('ls')")], escaped_help=[('?', 'get_ipython().show_usage()'), ('?x1', "get_ipython().run_line_magic('pinfo', 'x1')"), ('??x2', "get_ipython().run_line_magic('pinfo2', 'x2')"), ('?a.*s', "get_ipython().run_line_magic('psearch', 'a.*s')"), ('?%hist1', "get_ipython().run_line_magic('pinfo', '%hist1')"), ('?%%hist2', "get_ipython().run_line_magic('pinfo', '%%hist2')"), ('?abc = qwe', "get_ipython().run_line_magic('pinfo', 'abc')")], end_help=[('x3?', "get_ipython().run_line_magic('pinfo', 'x3')"), ('x4??', "get_ipython().run_line_magic('pinfo2', 'x4')"), ('%hist1?', "get_ipython().run_line_magic('pinfo', '%hist1')"), ('%hist2??', "get_ipython().run_line_magic('pinfo2', '%hist2')"), ('%%hist3?', "get_ipython().run_line_magic('pinfo', '%%hist3')"), ('%%hist4??', "get_ipython().run_line_magic('pinfo2', '%%hist4')"), ('π.foo?', "get_ipython().run_line_magic('pinfo', 'π.foo')"), ('f*?', "get_ipython().run_line_magic('psearch', 'f*')"), ('ax.*aspe*?', "get_ipython().run_line_magic('psearch', 'ax.*aspe*')"), ('a = abc?', "get_ipython().run_line_magic('pinfo', 'abc')"), ('a = abc.qe??', "get_ipython().run_line_magic('pinfo2', 'abc.qe')"), ('a = *.items?', "get_ipython().run_line_magic('psearch', '*.items')"), ('plot(a?', "get_ipython().run_line_magic('pinfo', 'a')"), ('a*2 #comment?', 'a*2 #comment?')], escaped_magic=[('%cd', "get_ipython().run_line_magic('cd', '')"), ('%cd /home', "get_ipython().run_line_magic('cd', '/home')"), ('%cd C:\\User', "get_ipython().run_line_magic('cd', 'C:\\\\User')"), ('    %magic', "    get_ipython().run_line_magic('magic', '')")], escaped_quote=[(',f', 'f("")'), (',f x', 'f("x")'), ('  ,f y', '  f("y")'), (',f a b', 'f("a", "b")')], escaped_quote2=[(';f', 'f("")'), (';f x', 'f("x")'), ('  ;f y', '  f("y")'), (';f a b', 'f("a b")')], escaped_paren=[('/f', 'f()'), ('/f x', 'f(x)'), ('  /f y', '  f(y)'), ('/f a b', 'f(a, b)')], mixed=[('In [1]: %lsmagic', "get_ipython().run_line_magic('lsmagic', '')"), ('>>> %lsmagic', "get_ipython().run_line_magic('lsmagic', '')"), ('In [2]: !ls', "get_ipython().system('ls')"), ('In [3]: abs?', "get_ipython().run_line_magic('pinfo', 'abs')"), ('In [4]: b = %who', "b = get_ipython().run_line_magic('who', '')")])
syntax_ml = dict(classic_prompt=[[('>>> for i in range(10):', 'for i in range(10):'), ('...     print i', '    print i'), ('... ', '')], [('>>> a="""', 'a="""'), ('... 123"""', '123"""')], [('a="""', 'a="""'), ('... 123', '123'), ('... 456"""', '456"""')], [('a="""', 'a="""'), ('>>> 123', '123'), ('... 456"""', '456"""')], [('a="""', 'a="""'), ('123', '123'), ('... 456"""', '... 456"""')], [('....__class__', '....__class__')], [('a=5', 'a=5'), ('...', '')], [('>>> def f(x):', 'def f(x):'), ('...', ''), ('...     return x', '    return x')], [('board = """....', 'board = """....'), ('....', '....'), ('...."""', '...."""')]], ipy_prompt=[[('In [24]: for i in range(10):', 'for i in range(10):'), ('   ....:     print i', '    print i'), ('   ....: ', '')], [('In [24]: for i in range(10):', 'for i in range(10):'), ('    ...:     print i', '    print i'), ('    ...: ', '')], [('In [24]: for i in range(10):', 'for i in range(10):'), ('...:     print i', '    print i'), ('...: ', '')], [('In [24]: for i in range(10):', 'for i in range(10):'), ('...:     print i', '    print i'), ('...:', '')], [('In [2]: a="""', 'a="""'), ('   ...: 123"""', '123"""')], [('a="""', 'a="""'), ('   ...: 123', '123'), ('   ...: 456"""', '456"""')], [('a="""', 'a="""'), ('In [1]: 123', '123'), ('   ...: 456"""', '456"""')], [('a="""', 'a="""'), ('123', '123'), ('   ...: 456"""', '   ...: 456"""')]], multiline_datastructure_prompt=[[('>>> a = [1,', 'a = [1,'), ('... 2]', '2]')]], multiline_datastructure=[[('b = ("%s"', None), ('# comment', None), ('%foo )', 'b = ("%s"\n# comment\n%foo )')]], multiline_string=[[("'''foo?", None), ("bar'''", "'''foo?\nbar'''")]], leading_indent=[[('    print "hi"', 'print "hi"')], [('  for a in range(5):', 'for a in range(5):'), ('    a*2', '  a*2')], [('    a="""', 'a="""'), ('    123"""', '123"""')], [('a="""', 'a="""'), ('    123"""', '    123"""')]], cellmagic=[[('%%foo a', None), (None, "get_ipython().run_cell_magic('foo', 'a', '')")], [('%%bar 123', None), ('hello', None), (None, "get_ipython().run_cell_magic('bar', '123', 'hello')")], [('a=5', 'a=5'), ('%%cellmagic', '%%cellmagic')]], escaped=[[('%abc def \\', None), ('ghi', "get_ipython().run_line_magic('abc', 'def ghi')")], [('%abc def \\', None), ('ghi\\', None), (None, "get_ipython().run_line_magic('abc', 'def ghi')")]], assign_magic=[[('a = %bc de \\', None), ('fg', "a = get_ipython().run_line_magic('bc', 'de fg')")], [('a = %bc de \\', None), ('fg\\', None), (None, "a = get_ipython().run_line_magic('bc', 'de fg')")]], assign_system=[[('a = !bc de \\', None), ('fg', "a = get_ipython().getoutput('bc de fg')")], [('a = !bc de \\', None), ('fg\\', None), (None, "a = get_ipython().getoutput('bc de fg')")]])

def test_assign_system():
    if False:
        for i in range(10):
            print('nop')
    tt.check_pairs(transform_and_reset(ipt.assign_from_system), syntax['assign_system'])

def test_assign_magic():
    if False:
        for i in range(10):
            print('nop')
    tt.check_pairs(transform_and_reset(ipt.assign_from_magic), syntax['assign_magic'])

def test_classic_prompt():
    if False:
        while True:
            i = 10
    tt.check_pairs(transform_and_reset(ipt.classic_prompt), syntax['classic_prompt'])
    for example in syntax_ml['classic_prompt']:
        transform_checker(example, ipt.classic_prompt)
    for example in syntax_ml['multiline_datastructure_prompt']:
        transform_checker(example, ipt.classic_prompt)
    transform_checker([('%foo', '%foo'), ('>>> bar', '>>> bar')], ipt.classic_prompt)

def test_ipy_prompt():
    if False:
        i = 10
        return i + 15
    tt.check_pairs(transform_and_reset(ipt.ipy_prompt), syntax['ipy_prompt'])
    for example in syntax_ml['ipy_prompt']:
        transform_checker(example, ipt.ipy_prompt)
    transform_checker([('%%foo', '%%foo'), ('In [1]: bar', 'In [1]: bar')], ipt.ipy_prompt)

def test_assemble_logical_lines():
    if False:
        print('Hello World!')
    tests = [[('a = \\', None), ('123', 'a = 123')], [('a = \\', None), ('12 *\\', None), (None, 'a = 12 *')], [('# foo\\', '# foo\\')]]
    for example in tests:
        transform_checker(example, ipt.assemble_logical_lines)

def test_assemble_python_lines():
    if False:
        i = 10
        return i + 15
    tests = [[("a = '''", None), ("abc'''", "a = '''\nabc'''")], [("a = '''", None), ('def', None), (None, "a = '''\ndef")], [('a = [1,', None), ('2]', 'a = [1,\n2]')], [('a = [1,', None), ('2,', None), (None, 'a = [1,\n2,')], [("a = '''", None), ('abc\\', None), ('def', None), ("'''", "a = '''\nabc\\\ndef\n'''")]] + syntax_ml['multiline_datastructure']
    for example in tests:
        transform_checker(example, ipt.assemble_python_lines)

def test_help_end():
    if False:
        while True:
            i = 10
    tt.check_pairs(transform_and_reset(ipt.help_end), syntax['end_help'])

def test_escaped_noesc():
    if False:
        while True:
            i = 10
    tt.check_pairs(transform_and_reset(ipt.escaped_commands), syntax['escaped_noesc'])

def test_escaped_shell():
    if False:
        for i in range(10):
            print('nop')
    tt.check_pairs(transform_and_reset(ipt.escaped_commands), syntax['escaped_shell'])

def test_escaped_help():
    if False:
        print('Hello World!')
    tt.check_pairs(transform_and_reset(ipt.escaped_commands), syntax['escaped_help'])

def test_escaped_magic():
    if False:
        for i in range(10):
            print('nop')
    tt.check_pairs(transform_and_reset(ipt.escaped_commands), syntax['escaped_magic'])

def test_escaped_quote():
    if False:
        i = 10
        return i + 15
    tt.check_pairs(transform_and_reset(ipt.escaped_commands), syntax['escaped_quote'])

def test_escaped_quote2():
    if False:
        i = 10
        return i + 15
    tt.check_pairs(transform_and_reset(ipt.escaped_commands), syntax['escaped_quote2'])

def test_escaped_paren():
    if False:
        print('Hello World!')
    tt.check_pairs(transform_and_reset(ipt.escaped_commands), syntax['escaped_paren'])

def test_cellmagic():
    if False:
        i = 10
        return i + 15
    for example in syntax_ml['cellmagic']:
        transform_checker(example, ipt.cellmagic)
    line_example = [('%%bar 123', None), ('hello', None), ('', "get_ipython().run_cell_magic('bar', '123', 'hello')")]
    transform_checker(line_example, ipt.cellmagic, end_on_blank_line=True)

def test_has_comment():
    if False:
        while True:
            i = 10
    tests = [('text', False), ('text #comment', True), ('text #comment\n', True), ('#comment', True), ('#comment\n', True), ('a = "#string"', False), ('a = "#string" # comment', True), ('a #comment not "string"', True)]
    tt.check_pairs(ipt.has_comment, tests)

@ipt.TokenInputTransformer.wrap
def decistmt(tokens):
    if False:
        for i in range(10):
            print('nop')
    'Substitute Decimals for floats in a string of statements.\n\n    Based on an example from the tokenize module docs.\n    '
    result = []
    for (toknum, tokval, _, _, _) in tokens:
        if toknum == tokenize.NUMBER and '.' in tokval:
            yield from [(tokenize.NAME, 'Decimal'), (tokenize.OP, '('), (tokenize.STRING, repr(tokval)), (tokenize.OP, ')')]
        else:
            yield (toknum, tokval)

def test_token_input_transformer():
    if False:
        while True:
            i = 10
    tests = [('1.2', "Decimal ('1.2')"), ('"1.2"', '"1.2"')]
    tt.check_pairs(transform_and_reset(decistmt), tests)
    ml_tests = [[("a = 1.2; b = '''x", None), ("y'''", "a =Decimal ('1.2');b ='''x\ny'''")], [('a = [1.2,', None), ('3]', "a =[Decimal ('1.2'),\n3 ]")], [("a = '''foo", None), ('bar', None), (None, "a = '''foo\nbar")]]
    for example in ml_tests:
        transform_checker(example, decistmt)