from os import walk, sep, pardir
from os.path import split, join, abspath, exists, isfile
from glob import glob
import re
import random
import ast
from sympy.testing.pytest import raises
from sympy.testing.quality_unicode import _test_this_file_encoding
sepd = {'sep': sep}
SYMPY_PATH = abspath(join(split(__file__)[0], pardir, pardir))
assert exists(SYMPY_PATH)
TOP_PATH = abspath(join(SYMPY_PATH, pardir))
BIN_PATH = join(TOP_PATH, 'bin')
EXAMPLES_PATH = join(TOP_PATH, 'examples')
message_space = 'File contains trailing whitespace: %s, line %s.'
message_implicit = 'File contains an implicit import: %s, line %s.'
message_tabs = 'File contains tabs instead of spaces: %s, line %s.'
message_carriage = 'File contains carriage returns at end of line: %s, line %s'
message_str_raise = 'File contains string exception: %s, line %s'
message_gen_raise = 'File contains generic exception: %s, line %s'
message_old_raise = 'File contains old-style raise statement: %s, line %s, "%s"'
message_eof = 'File does not end with a newline: %s, line %s'
message_multi_eof = 'File ends with more than 1 newline: %s, line %s'
message_test_suite_def = "Function should start with 'test_' or '_': %s, line %s"
message_duplicate_test = 'This is a duplicate test function: %s, line %s'
message_self_assignments = 'File contains assignments to self/cls: %s, line %s.'
message_func_is = "File contains '.func is': %s, line %s."
message_bare_expr = 'File contains bare expression: %s, line %s.'
implicit_test_re = re.compile('^\\s*(>>> )?(\\.\\.\\. )?from .* import .*\\*')
str_raise_re = re.compile('^\\s*(>>> )?(\\.\\.\\. )?raise(\\s+(\\\'|\\")|\\s*(\\(\\s*)+(\\\'|\\"))')
gen_raise_re = re.compile('^\\s*(>>> )?(\\.\\.\\. )?raise(\\s+Exception|\\s*(\\(\\s*)+Exception)')
old_raise_re = re.compile('^\\s*(>>> )?(\\.\\.\\. )?raise((\\s*\\(\\s*)|\\s+)\\w+\\s*,')
test_suite_def_re = re.compile('^def\\s+(?!(_|test))[^(]*\\(\\s*\\)\\s*:$')
test_ok_def_re = re.compile('^def\\s+test_.*:$')
test_file_re = re.compile('.*[/\\\\]test_.*\\.py$')
func_is_re = re.compile('\\.\\s*func\\s+is')

def tab_in_leading(s):
    if False:
        i = 10
        return i + 15
    'Returns True if there are tabs in the leading whitespace of a line,\n    including the whitespace of docstring code samples.'
    n = len(s) - len(s.lstrip())
    if not s[n:n + 3] in ['...', '>>>']:
        check = s[:n]
    else:
        smore = s[n + 3:]
        check = s[:n] + smore[:len(smore) - len(smore.lstrip())]
    return not check.expandtabs() == check

def find_self_assignments(s):
    if False:
        print('Hello World!')
    'Returns a list of "bad" assignments: if there are instances\n    of assigning to the first argument of the class method (except\n    for staticmethod\'s).\n    '
    t = [n for n in ast.parse(s).body if isinstance(n, ast.ClassDef)]
    bad = []
    for c in t:
        for n in c.body:
            if not isinstance(n, ast.FunctionDef):
                continue
            if any((d.id == 'staticmethod' for d in n.decorator_list if isinstance(d, ast.Name))):
                continue
            if n.name == '__new__':
                continue
            if not n.args.args:
                continue
            first_arg = n.args.args[0].arg
            for m in ast.walk(n):
                if isinstance(m, ast.Assign):
                    for a in m.targets:
                        if isinstance(a, ast.Name) and a.id == first_arg:
                            bad.append(m)
                        elif isinstance(a, ast.Tuple) and any((q.id == first_arg for q in a.elts if isinstance(q, ast.Name))):
                            bad.append(m)
    return bad

def check_directory_tree(base_path, file_check, exclusions=set(), pattern='*.py'):
    if False:
        print('Hello World!')
    '\n    Checks all files in the directory tree (with base_path as starting point)\n    with the file_check function provided, skipping files that contain\n    any of the strings in the set provided by exclusions.\n    '
    if not base_path:
        return
    for (root, dirs, files) in walk(base_path):
        check_files(glob(join(root, pattern)), file_check, exclusions)

def check_files(files, file_check, exclusions=set(), pattern=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Checks all files with the file_check function provided, skipping files\n    that contain any of the strings in the set provided by exclusions.\n    '
    if not files:
        return
    for fname in files:
        if not exists(fname) or not isfile(fname):
            continue
        if any((ex in fname for ex in exclusions)):
            continue
        if pattern is None or re.match(pattern, fname):
            file_check(fname)

class _Visit(ast.NodeVisitor):
    """return the line number corresponding to the
    line on which a bare expression appears if it is a binary op
    or a comparison that is not in a with block.

    EXAMPLES
    ========

    >>> import ast
    >>> class _Visit(ast.NodeVisitor):
    ...     def visit_Expr(self, node):
    ...         if isinstance(node.value, (ast.BinOp, ast.Compare)):
    ...             print(node.lineno)
    ...     def visit_With(self, node):
    ...         pass  # no checking there
    ...
    >>> code='''x = 1    # line 1
    ... for i in range(3):
    ...     x == 2       # <-- 3
    ... if x == 2:
    ...     x == 3       # <-- 5
    ...     x + 1        # <-- 6
    ...     x = 1
    ...     if x == 1:
    ...         print(1)
    ... while x != 1:
    ...     x == 1       # <-- 11
    ... with raises(TypeError):
    ...     c == 1
    ...     raise TypeError
    ... assert x == 1
    ... '''
    >>> _Visit().visit(ast.parse(code))
    3
    5
    6
    11
    """

    def visit_Expr(self, node):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(node.value, (ast.BinOp, ast.Compare)):
            assert None, message_bare_expr % ('', node.lineno)

    def visit_With(self, node):
        if False:
            return 10
        pass
BareExpr = _Visit()

def line_with_bare_expr(code):
    if False:
        print('Hello World!')
    'return None or else 0-based line number of code on which\n    a bare expression appeared.\n    '
    tree = ast.parse(code)
    try:
        BareExpr.visit(tree)
    except AssertionError as msg:
        assert msg.args
        msg = msg.args[0]
        assert msg.startswith(message_bare_expr.split(':', 1)[0])
        return int(msg.rsplit(' ', 1)[1].rstrip('.'))

def test_files():
    if False:
        for i in range(10):
            print('nop')
    '\n    This test tests all files in SymPy and checks that:\n      o no lines contains a trailing whitespace\n      o no lines end with \r\n\n      o no line uses tabs instead of spaces\n      o that the file ends with a single newline\n      o there are no general or string exceptions\n      o there are no old style raise statements\n      o name of arg-less test suite functions start with _ or test_\n      o no duplicate function names that start with test_\n      o no assignments to self variable in class methods\n      o no lines contain ".func is" except in the test suite\n      o there is no do-nothing expression like `a == b` or `x + 1`\n    '

    def test(fname):
        if False:
            i = 10
            return i + 15
        with open(fname, encoding='utf8') as test_file:
            test_this_file(fname, test_file)
        with open(fname, encoding='utf8') as test_file:
            _test_this_file_encoding(fname, test_file)

    def test_this_file(fname, test_file):
        if False:
            print('Hello World!')
        idx = None
        code = test_file.read()
        test_file.seek(0)
        py = fname if sep not in fname else fname.rsplit(sep, 1)[-1]
        if py.startswith('test_'):
            idx = line_with_bare_expr(code)
        if idx is not None:
            assert False, message_bare_expr % (fname, idx + 1)
        line = None
        tests = 0
        test_set = set()
        for (idx, line) in enumerate(test_file):
            if test_file_re.match(fname):
                if test_suite_def_re.match(line):
                    assert False, message_test_suite_def % (fname, idx + 1)
                if test_ok_def_re.match(line):
                    tests += 1
                    test_set.add(line[3:].split('(')[0].strip())
                    if len(test_set) != tests:
                        assert False, message_duplicate_test % (fname, idx + 1)
            if line.endswith((' \n', '\t\n')):
                assert False, message_space % (fname, idx + 1)
            if line.endswith('\r\n'):
                assert False, message_carriage % (fname, idx + 1)
            if tab_in_leading(line):
                assert False, message_tabs % (fname, idx + 1)
            if str_raise_re.search(line):
                assert False, message_str_raise % (fname, idx + 1)
            if gen_raise_re.search(line):
                assert False, message_gen_raise % (fname, idx + 1)
            if implicit_test_re.search(line) and (not list(filter(lambda ex: ex in fname, import_exclude))):
                assert False, message_implicit % (fname, idx + 1)
            if func_is_re.search(line) and (not test_file_re.search(fname)):
                assert False, message_func_is % (fname, idx + 1)
            result = old_raise_re.search(line)
            if result is not None:
                assert False, message_old_raise % (fname, idx + 1, result.group(2))
        if line is not None:
            if line == '\n' and idx > 0:
                assert False, message_multi_eof % (fname, idx + 1)
            elif not line.endswith('\n'):
                assert False, message_eof % (fname, idx + 1)
    top_level_files = [join(TOP_PATH, file) for file in ['isympy.py', 'build.py', 'setup.py']]
    exclude = {'%(sep)ssympy%(sep)sparsing%(sep)sautolev%(sep)s_antlr%(sep)sautolevparser.py' % sepd, '%(sep)ssympy%(sep)sparsing%(sep)sautolev%(sep)s_antlr%(sep)sautolevlexer.py' % sepd, '%(sep)ssympy%(sep)sparsing%(sep)sautolev%(sep)s_antlr%(sep)sautolevlistener.py' % sepd, '%(sep)ssympy%(sep)sparsing%(sep)slatex%(sep)s_antlr%(sep)slatexparser.py' % sepd, '%(sep)ssympy%(sep)sparsing%(sep)slatex%(sep)s_antlr%(sep)slatexlexer.py' % sepd}
    import_exclude = {'%(sep)ssympy%(sep)s__init__.py' % sepd, '%(sep)svector%(sep)s__init__.py' % sepd, '%(sep)smechanics%(sep)s__init__.py' % sepd, '%(sep)squantum%(sep)s__init__.py' % sepd, '%(sep)spolys%(sep)s__init__.py' % sepd, '%(sep)spolys%(sep)sdomains%(sep)s__init__.py' % sepd, '%(sep)sinteractive%(sep)ssession.py' % sepd, '%(sep)sisympy.py' % sepd, '%(sep)sbin%(sep)ssympy_time.py' % sepd, '%(sep)sbin%(sep)ssympy_time_cache.py' % sepd, '%(sep)sparsing%(sep)ssympy_tokenize.py' % sepd, '%(sep)splotting%(sep)spygletplot%(sep)s' % sepd, '%(sep)sbin%(sep)stest_external_imports.py' % sepd, '%(sep)sbin%(sep)stest_submodule_imports.py' % sepd, '%(sep)sutilities%(sep)sruntests.py' % sepd, '%(sep)sutilities%(sep)spytest.py' % sepd, '%(sep)sutilities%(sep)srandtest.py' % sepd, '%(sep)sutilities%(sep)stmpfiles.py' % sepd, '%(sep)sutilities%(sep)squality_unicode.py' % sepd}
    check_files(top_level_files, test)
    check_directory_tree(BIN_PATH, test, {'~', '.pyc', '.sh', '.mjs'}, '*')
    check_directory_tree(SYMPY_PATH, test, exclude)
    check_directory_tree(EXAMPLES_PATH, test, exclude)

def _with_space(c):
    if False:
        i = 10
        return i + 15
    return random.randint(0, 10) * ' ' + c

def test_raise_statement_regular_expression():
    if False:
        while True:
            i = 10
    candidates_ok = ["some text # raise Exception, 'text'", "raise ValueError('text') # raise Exception, 'text'", "raise ValueError('text')", 'raise ValueError', "raise ValueError('text')", "raise ValueError('text') #,", '\'"""This function will raise ValueError, except when it doesn\'t"""', "raise (ValueError('text')"]
    str_candidates_fail = ["raise 'exception'", "raise 'Exception'", 'raise "exception"', 'raise "Exception"', "raise 'ValueError'"]
    gen_candidates_fail = ["raise Exception('text') # raise Exception, 'text'", "raise Exception('text')", 'raise Exception', "raise Exception('text')", "raise Exception('text') #,", "raise Exception, 'text'", "raise Exception, 'text' # raise Exception('text')", "raise Exception, 'text' # raise Exception, 'text'", ">>> raise Exception, 'text'", ">>> raise Exception, 'text' # raise Exception('text')", ">>> raise Exception, 'text' # raise Exception, 'text'"]
    old_candidates_fail = ["raise Exception, 'text'", "raise Exception, 'text' # raise Exception('text')", "raise Exception, 'text' # raise Exception, 'text'", ">>> raise Exception, 'text'", ">>> raise Exception, 'text' # raise Exception('text')", ">>> raise Exception, 'text' # raise Exception, 'text'", "raise ValueError, 'text'", "raise ValueError, 'text' # raise Exception('text')", "raise ValueError, 'text' # raise Exception, 'text'", ">>> raise ValueError, 'text'", ">>> raise ValueError, 'text' # raise Exception('text')", ">>> raise ValueError, 'text' # raise Exception, 'text'", 'raise(ValueError,', 'raise (ValueError,', 'raise( ValueError,', 'raise ( ValueError,', 'raise(ValueError ,', 'raise (ValueError ,', 'raise( ValueError ,', 'raise ( ValueError ,']
    for c in candidates_ok:
        assert str_raise_re.search(_with_space(c)) is None, c
        assert gen_raise_re.search(_with_space(c)) is None, c
        assert old_raise_re.search(_with_space(c)) is None, c
    for c in str_candidates_fail:
        assert str_raise_re.search(_with_space(c)) is not None, c
    for c in gen_candidates_fail:
        assert gen_raise_re.search(_with_space(c)) is not None, c
    for c in old_candidates_fail:
        assert old_raise_re.search(_with_space(c)) is not None, c

def test_implicit_imports_regular_expression():
    if False:
        return 10
    candidates_ok = ['from sympy import something', '>>> from sympy import something', 'from sympy.somewhere import something', '>>> from sympy.somewhere import something', 'import sympy', '>>> import sympy', 'import sympy.something.something', '... import sympy', '... import sympy.something.something', '... from sympy import something', '... from sympy.somewhere import something', '>> from sympy import *', '# from sympy import *', 'some text # from sympy import *']
    candidates_fail = ['from sympy import *', '>>> from sympy import *', 'from sympy.somewhere import *', '>>> from sympy.somewhere import *', '... from sympy import *', '... from sympy.somewhere import *']
    for c in candidates_ok:
        assert implicit_test_re.search(_with_space(c)) is None, c
    for c in candidates_fail:
        assert implicit_test_re.search(_with_space(c)) is not None, c

def test_test_suite_defs():
    if False:
        return 10
    candidates_ok = ['    def foo():\n', 'def foo(arg):\n', 'def _foo():\n', 'def test_foo():\n']
    candidates_fail = ['def foo():\n', 'def foo() :\n', 'def foo( ):\n', 'def  foo():\n']
    for c in candidates_ok:
        assert test_suite_def_re.search(c) is None, c
    for c in candidates_fail:
        assert test_suite_def_re.search(c) is not None, c

def test_test_duplicate_defs():
    if False:
        return 10
    candidates_ok = ['def foo():\ndef foo():\n', 'def test():\ndef test_():\n', 'def test_():\ndef test__():\n']
    candidates_fail = ['def test_():\ndef test_ ():\n', 'def test_1():\ndef  test_1():\n']
    ok = (None, 'check')

    def check(file):
        if False:
            for i in range(10):
                print('nop')
        tests = 0
        test_set = set()
        for (idx, line) in enumerate(file.splitlines()):
            if test_ok_def_re.match(line):
                tests += 1
                test_set.add(line[3:].split('(')[0].strip())
                if len(test_set) != tests:
                    return (False, message_duplicate_test % ('check', idx + 1))
        return (None, 'check')
    for c in candidates_ok:
        assert check(c) == ok
    for c in candidates_fail:
        assert check(c) != ok

def test_find_self_assignments():
    if False:
        print('Hello World!')
    candidates_ok = ['class A(object):\n    def foo(self, arg): arg = self\n', 'class A(object):\n    def foo(self, arg): self.prop = arg\n', 'class A(object):\n    def foo(self, arg): obj, obj2 = arg, self\n', 'class A(object):\n    @classmethod\n    def bar(cls, arg): arg = cls\n', 'class A(object):\n    def foo(var, arg): arg = var\n']
    candidates_fail = ['class A(object):\n    def foo(self, arg): self = arg\n', 'class A(object):\n    def foo(self, arg): obj, self = arg, arg\n', 'class A(object):\n    def foo(self, arg):\n        if arg: self = arg', 'class A(object):\n    @classmethod\n    def foo(cls, arg): cls = arg\n', 'class A(object):\n    def foo(var, arg): var = arg\n']
    for c in candidates_ok:
        assert find_self_assignments(c) == []
    for c in candidates_fail:
        assert find_self_assignments(c) != []

def test_test_unicode_encoding():
    if False:
        print('Hello World!')
    unicode_whitelist = ['foo']
    unicode_strict_whitelist = ['bar']
    fname = 'abc'
    test_file = ['α']
    raises(AssertionError, lambda : _test_this_file_encoding(fname, test_file, unicode_whitelist, unicode_strict_whitelist))
    fname = 'abc'
    test_file = ['abc']
    _test_this_file_encoding(fname, test_file, unicode_whitelist, unicode_strict_whitelist)
    fname = 'foo'
    test_file = ['abc']
    raises(AssertionError, lambda : _test_this_file_encoding(fname, test_file, unicode_whitelist, unicode_strict_whitelist))
    fname = 'bar'
    test_file = ['abc']
    _test_this_file_encoding(fname, test_file, unicode_whitelist, unicode_strict_whitelist)