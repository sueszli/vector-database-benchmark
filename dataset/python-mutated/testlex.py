import unittest
try:
    import StringIO
except ImportError:
    import io as StringIO
import sys
import os
import warnings
import platform
sys.path.insert(0, '..')
sys.tracebacklimit = 0
import ply.lex
try:
    from importlib.util import cache_from_source
except ImportError:
    cache_from_source = None

def make_pymodule_path(filename, optimization=None):
    if False:
        i = 10
        return i + 15
    path = os.path.dirname(filename)
    file = os.path.basename(filename)
    (mod, ext) = os.path.splitext(file)
    if sys.hexversion >= 50659328:
        fullpath = cache_from_source(filename, optimization=optimization)
    elif sys.hexversion >= 50593792:
        fullpath = cache_from_source(filename, ext == '.pyc')
    elif sys.hexversion >= 50462720:
        import imp
        modname = mod + '.' + imp.get_tag() + ext
        fullpath = os.path.join(path, '__pycache__', modname)
    else:
        fullpath = filename
    return fullpath

def pymodule_out_exists(filename, optimization=None):
    if False:
        for i in range(10):
            print('nop')
    return os.path.exists(make_pymodule_path(filename, optimization=optimization))

def pymodule_out_remove(filename, optimization=None):
    if False:
        print('Hello World!')
    os.remove(make_pymodule_path(filename, optimization=optimization))

def implementation():
    if False:
        i = 10
        return i + 15
    if platform.system().startswith('Java'):
        return 'Jython'
    elif hasattr(sys, 'pypy_version_info'):
        return 'PyPy'
    else:
        return 'CPython'
test_pyo = implementation() == 'CPython'

def check_expected(result, expected, contains=False):
    if False:
        for i in range(10):
            print('nop')
    if sys.version_info[0] >= 3:
        if isinstance(result, str):
            result = result.encode('ascii')
        if isinstance(expected, str):
            expected = expected.encode('ascii')
    resultlines = result.splitlines()
    expectedlines = expected.splitlines()
    if len(resultlines) != len(expectedlines):
        return False
    for (rline, eline) in zip(resultlines, expectedlines):
        if contains:
            if eline not in rline:
                return False
        elif not rline.endswith(eline):
            return False
    return True

def run_import(module):
    if False:
        for i in range(10):
            print('nop')
    code = 'import ' + module
    exec(code)
    del sys.modules[module]

class LexErrorWarningTests(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        sys.stderr = StringIO.StringIO()
        sys.stdout = StringIO.StringIO()
        if sys.hexversion >= 50462720:
            warnings.filterwarnings('ignore', category=ResourceWarning)

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        sys.stderr = sys.__stderr__
        sys.stdout = sys.__stdout__

    def test_lex_doc1(self):
        if False:
            return 10
        self.assertRaises(SyntaxError, run_import, 'lex_doc1')
        result = sys.stderr.getvalue()
        self.assert_(check_expected(result, "lex_doc1.py:18: No regular expression defined for rule 't_NUMBER'\n"))

    def test_lex_dup1(self):
        if False:
            print('Hello World!')
        self.assertRaises(SyntaxError, run_import, 'lex_dup1')
        result = sys.stderr.getvalue()
        self.assert_(check_expected(result, 'lex_dup1.py:20: Rule t_NUMBER redefined. Previously defined on line 18\n'))

    def test_lex_dup2(self):
        if False:
            return 10
        self.assertRaises(SyntaxError, run_import, 'lex_dup2')
        result = sys.stderr.getvalue()
        self.assert_(check_expected(result, 'lex_dup2.py:22: Rule t_NUMBER redefined. Previously defined on line 18\n'))

    def test_lex_dup3(self):
        if False:
            print('Hello World!')
        self.assertRaises(SyntaxError, run_import, 'lex_dup3')
        result = sys.stderr.getvalue()
        self.assert_(check_expected(result, 'lex_dup3.py:20: Rule t_NUMBER redefined. Previously defined on line 18\n'))

    def test_lex_empty(self):
        if False:
            print('Hello World!')
        self.assertRaises(SyntaxError, run_import, 'lex_empty')
        result = sys.stderr.getvalue()
        self.assert_(check_expected(result, "No rules of the form t_rulename are defined\nNo rules defined for state 'INITIAL'\n"))

    def test_lex_error1(self):
        if False:
            i = 10
            return i + 15
        run_import('lex_error1')
        result = sys.stderr.getvalue()
        self.assert_(check_expected(result, 'No t_error rule is defined\n'))

    def test_lex_error2(self):
        if False:
            while True:
                i = 10
        self.assertRaises(SyntaxError, run_import, 'lex_error2')
        result = sys.stderr.getvalue()
        self.assert_(check_expected(result, "Rule 't_error' must be defined as a function\n"))

    def test_lex_error3(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertRaises(SyntaxError, run_import, 'lex_error3')
        result = sys.stderr.getvalue()
        self.assert_(check_expected(result, "lex_error3.py:20: Rule 't_error' requires an argument\n"))

    def test_lex_error4(self):
        if False:
            i = 10
            return i + 15
        self.assertRaises(SyntaxError, run_import, 'lex_error4')
        result = sys.stderr.getvalue()
        self.assert_(check_expected(result, "lex_error4.py:20: Rule 't_error' has too many arguments\n"))

    def test_lex_ignore(self):
        if False:
            while True:
                i = 10
        self.assertRaises(SyntaxError, run_import, 'lex_ignore')
        result = sys.stderr.getvalue()
        self.assert_(check_expected(result, "lex_ignore.py:20: Rule 't_ignore' must be defined as a string\n"))

    def test_lex_ignore2(self):
        if False:
            return 10
        run_import('lex_ignore2')
        result = sys.stderr.getvalue()
        self.assert_(check_expected(result, "t_ignore contains a literal backslash '\\'\n"))

    def test_lex_re1(self):
        if False:
            return 10
        self.assertRaises(SyntaxError, run_import, 'lex_re1')
        result = sys.stderr.getvalue()
        if sys.hexversion < 50659328:
            msg = "Invalid regular expression for rule 't_NUMBER'. unbalanced parenthesis\n"
        else:
            msg = "Invalid regular expression for rule 't_NUMBER'. missing ), unterminated subpattern at position 0"
        self.assert_(check_expected(result, msg, contains=True))

    def test_lex_re2(self):
        if False:
            print('Hello World!')
        self.assertRaises(SyntaxError, run_import, 'lex_re2')
        result = sys.stderr.getvalue()
        self.assert_(check_expected(result, "Regular expression for rule 't_PLUS' matches empty string\n"))

    def test_lex_re3(self):
        if False:
            return 10
        self.assertRaises(SyntaxError, run_import, 'lex_re3')
        result = sys.stderr.getvalue()
        if sys.hexversion < 50659328:
            msg = "Invalid regular expression for rule 't_POUND'. unbalanced parenthesis\nMake sure '#' in rule 't_POUND' is escaped with '\\#'\n"
        else:
            msg = "Invalid regular expression for rule 't_POUND'. missing ), unterminated subpattern at position 0\nERROR: Make sure '#' in rule 't_POUND' is escaped with '\\#'"
        self.assert_(check_expected(result, msg, contains=True), result)

    def test_lex_rule1(self):
        if False:
            while True:
                i = 10
        self.assertRaises(SyntaxError, run_import, 'lex_rule1')
        result = sys.stderr.getvalue()
        self.assert_(check_expected(result, 't_NUMBER not defined as a function or string\n'))

    def test_lex_rule2(self):
        if False:
            while True:
                i = 10
        self.assertRaises(SyntaxError, run_import, 'lex_rule2')
        result = sys.stderr.getvalue()
        self.assert_(check_expected(result, "lex_rule2.py:18: Rule 't_NUMBER' requires an argument\n"))

    def test_lex_rule3(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertRaises(SyntaxError, run_import, 'lex_rule3')
        result = sys.stderr.getvalue()
        self.assert_(check_expected(result, "lex_rule3.py:18: Rule 't_NUMBER' has too many arguments\n"))

    def test_lex_state1(self):
        if False:
            i = 10
            return i + 15
        self.assertRaises(SyntaxError, run_import, 'lex_state1')
        result = sys.stderr.getvalue()
        self.assert_(check_expected(result, 'states must be defined as a tuple or list\n'))

    def test_lex_state2(self):
        if False:
            while True:
                i = 10
        self.assertRaises(SyntaxError, run_import, 'lex_state2')
        result = sys.stderr.getvalue()
        self.assert_(check_expected(result, "Invalid state specifier 'comment'. Must be a tuple (statename,'exclusive|inclusive')\nInvalid state specifier 'example'. Must be a tuple (statename,'exclusive|inclusive')\n"))

    def test_lex_state3(self):
        if False:
            print('Hello World!')
        self.assertRaises(SyntaxError, run_import, 'lex_state3')
        result = sys.stderr.getvalue()
        self.assert_(check_expected(result, "State name 1 must be a string\nNo rules defined for state 'example'\n"))

    def test_lex_state4(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertRaises(SyntaxError, run_import, 'lex_state4')
        result = sys.stderr.getvalue()
        self.assert_(check_expected(result, "State type for state comment must be 'inclusive' or 'exclusive'\n"))

    def test_lex_state5(self):
        if False:
            i = 10
            return i + 15
        self.assertRaises(SyntaxError, run_import, 'lex_state5')
        result = sys.stderr.getvalue()
        self.assert_(check_expected(result, "State 'comment' already defined\n"))

    def test_lex_state_noerror(self):
        if False:
            return 10
        run_import('lex_state_noerror')
        result = sys.stderr.getvalue()
        self.assert_(check_expected(result, "No error rule is defined for exclusive state 'comment'\n"))

    def test_lex_state_norule(self):
        if False:
            print('Hello World!')
        self.assertRaises(SyntaxError, run_import, 'lex_state_norule')
        result = sys.stderr.getvalue()
        self.assert_(check_expected(result, "No rules defined for state 'example'\n"))

    def test_lex_token1(self):
        if False:
            return 10
        self.assertRaises(SyntaxError, run_import, 'lex_token1')
        result = sys.stderr.getvalue()
        self.assert_(check_expected(result, "No token list is defined\nRule 't_NUMBER' defined for an unspecified token NUMBER\nRule 't_PLUS' defined for an unspecified token PLUS\nRule 't_MINUS' defined for an unspecified token MINUS\n"))

    def test_lex_token2(self):
        if False:
            return 10
        self.assertRaises(SyntaxError, run_import, 'lex_token2')
        result = sys.stderr.getvalue()
        self.assert_(check_expected(result, "tokens must be a list or tuple\nRule 't_NUMBER' defined for an unspecified token NUMBER\nRule 't_PLUS' defined for an unspecified token PLUS\nRule 't_MINUS' defined for an unspecified token MINUS\n"))

    def test_lex_token3(self):
        if False:
            while True:
                i = 10
        self.assertRaises(SyntaxError, run_import, 'lex_token3')
        result = sys.stderr.getvalue()
        self.assert_(check_expected(result, "Rule 't_MINUS' defined for an unspecified token MINUS\n"))

    def test_lex_token4(self):
        if False:
            print('Hello World!')
        self.assertRaises(SyntaxError, run_import, 'lex_token4')
        result = sys.stderr.getvalue()
        self.assert_(check_expected(result, "Bad token name '-'\n"))

    def test_lex_token5(self):
        if False:
            while True:
                i = 10
        try:
            run_import('lex_token5')
        except ply.lex.LexError:
            e = sys.exc_info()[1]
        self.assert_(check_expected(str(e), "lex_token5.py:19: Rule 't_NUMBER' returned an unknown token type 'NUM'"))

    def test_lex_token_dup(self):
        if False:
            print('Hello World!')
        run_import('lex_token_dup')
        result = sys.stderr.getvalue()
        self.assert_(check_expected(result, "Token 'MINUS' multiply defined\n"))

    def test_lex_literal1(self):
        if False:
            return 10
        self.assertRaises(SyntaxError, run_import, 'lex_literal1')
        result = sys.stderr.getvalue()
        self.assert_(check_expected(result, "Invalid literal '**'. Must be a single character\n"))

    def test_lex_literal2(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertRaises(SyntaxError, run_import, 'lex_literal2')
        result = sys.stderr.getvalue()
        self.assert_(check_expected(result, 'Invalid literals specification. literals must be a sequence of characters\n'))
import os
import subprocess
import shutil

class LexBuildOptionTests(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        sys.stderr = StringIO.StringIO()
        sys.stdout = StringIO.StringIO()

    def tearDown(self):
        if False:
            while True:
                i = 10
        sys.stderr = sys.__stderr__
        sys.stdout = sys.__stdout__
        try:
            shutil.rmtree('lexdir')
        except OSError:
            pass

    def test_lex_module(self):
        if False:
            i = 10
            return i + 15
        run_import('lex_module')
        result = sys.stdout.getvalue()
        self.assert_(check_expected(result, "(NUMBER,3,1,0)\n(PLUS,'+',1,1)\n(NUMBER,4,1,2)\n"))

    def test_lex_object(self):
        if False:
            for i in range(10):
                print('nop')
        run_import('lex_object')
        result = sys.stdout.getvalue()
        self.assert_(check_expected(result, "(NUMBER,3,1,0)\n(PLUS,'+',1,1)\n(NUMBER,4,1,2)\n"))

    def test_lex_closure(self):
        if False:
            return 10
        run_import('lex_closure')
        result = sys.stdout.getvalue()
        self.assert_(check_expected(result, "(NUMBER,3,1,0)\n(PLUS,'+',1,1)\n(NUMBER,4,1,2)\n"))

    def test_lex_optimize(self):
        if False:
            i = 10
            return i + 15
        try:
            os.remove('lextab.py')
        except OSError:
            pass
        try:
            os.remove('lextab.pyc')
        except OSError:
            pass
        try:
            os.remove('lextab.pyo')
        except OSError:
            pass
        run_import('lex_optimize')
        result = sys.stdout.getvalue()
        self.assert_(check_expected(result, "(NUMBER,3,1,0)\n(PLUS,'+',1,1)\n(NUMBER,4,1,2)\n"))
        self.assert_(os.path.exists('lextab.py'))
        p = subprocess.Popen([sys.executable, '-O', 'lex_optimize.py'], stdout=subprocess.PIPE)
        result = p.stdout.read()
        self.assert_(check_expected(result, "(NUMBER,3,1,0)\n(PLUS,'+',1,1)\n(NUMBER,4,1,2)\n"))
        if test_pyo:
            self.assert_(pymodule_out_exists('lextab.pyo', 1))
            pymodule_out_remove('lextab.pyo', 1)
        p = subprocess.Popen([sys.executable, '-OO', 'lex_optimize.py'], stdout=subprocess.PIPE)
        result = p.stdout.read()
        self.assert_(check_expected(result, "(NUMBER,3,1,0)\n(PLUS,'+',1,1)\n(NUMBER,4,1,2)\n"))
        if test_pyo:
            self.assert_(pymodule_out_exists('lextab.pyo', 2))
        try:
            os.remove('lextab.py')
        except OSError:
            pass
        try:
            pymodule_out_remove('lextab.pyc')
        except OSError:
            pass
        try:
            pymodule_out_remove('lextab.pyo', 2)
        except OSError:
            pass

    def test_lex_optimize2(self):
        if False:
            i = 10
            return i + 15
        try:
            os.remove('opt2tab.py')
        except OSError:
            pass
        try:
            os.remove('opt2tab.pyc')
        except OSError:
            pass
        try:
            os.remove('opt2tab.pyo')
        except OSError:
            pass
        run_import('lex_optimize2')
        result = sys.stdout.getvalue()
        self.assert_(check_expected(result, "(NUMBER,3,1,0)\n(PLUS,'+',1,1)\n(NUMBER,4,1,2)\n"))
        self.assert_(os.path.exists('opt2tab.py'))
        p = subprocess.Popen([sys.executable, '-O', 'lex_optimize2.py'], stdout=subprocess.PIPE)
        result = p.stdout.read()
        self.assert_(check_expected(result, "(NUMBER,3,1,0)\n(PLUS,'+',1,1)\n(NUMBER,4,1,2)\n"))
        if test_pyo:
            self.assert_(pymodule_out_exists('opt2tab.pyo', 1))
            pymodule_out_remove('opt2tab.pyo', 1)
        p = subprocess.Popen([sys.executable, '-OO', 'lex_optimize2.py'], stdout=subprocess.PIPE)
        result = p.stdout.read()
        self.assert_(check_expected(result, "(NUMBER,3,1,0)\n(PLUS,'+',1,1)\n(NUMBER,4,1,2)\n"))
        if test_pyo:
            self.assert_(pymodule_out_exists('opt2tab.pyo', 2))
        try:
            os.remove('opt2tab.py')
        except OSError:
            pass
        try:
            pymodule_out_remove('opt2tab.pyc')
        except OSError:
            pass
        try:
            pymodule_out_remove('opt2tab.pyo', 2)
        except OSError:
            pass

    def test_lex_optimize3(self):
        if False:
            print('Hello World!')
        try:
            shutil.rmtree('lexdir')
        except OSError:
            pass
        os.mkdir('lexdir')
        os.mkdir('lexdir/sub')
        with open('lexdir/__init__.py', 'w') as f:
            f.write('')
        with open('lexdir/sub/__init__.py', 'w') as f:
            f.write('')
        run_import('lex_optimize3')
        result = sys.stdout.getvalue()
        self.assert_(check_expected(result, "(NUMBER,3,1,0)\n(PLUS,'+',1,1)\n(NUMBER,4,1,2)\n"))
        self.assert_(os.path.exists('lexdir/sub/calctab.py'))
        p = subprocess.Popen([sys.executable, '-O', 'lex_optimize3.py'], stdout=subprocess.PIPE)
        result = p.stdout.read()
        self.assert_(check_expected(result, "(NUMBER,3,1,0)\n(PLUS,'+',1,1)\n(NUMBER,4,1,2)\n"))
        if test_pyo:
            self.assert_(pymodule_out_exists('lexdir/sub/calctab.pyo', 1))
            pymodule_out_remove('lexdir/sub/calctab.pyo', 1)
        p = subprocess.Popen([sys.executable, '-OO', 'lex_optimize3.py'], stdout=subprocess.PIPE)
        result = p.stdout.read()
        self.assert_(check_expected(result, "(NUMBER,3,1,0)\n(PLUS,'+',1,1)\n(NUMBER,4,1,2)\n"))
        if test_pyo:
            self.assert_(pymodule_out_exists('lexdir/sub/calctab.pyo', 2))
        try:
            shutil.rmtree('lexdir')
        except OSError:
            pass

    def test_lex_optimize4(self):
        if False:
            i = 10
            return i + 15
        for extension in ['py', 'pyc']:
            try:
                os.remove('opt4tab.{0}'.format(extension))
            except OSError:
                pass
        run_import('lex_optimize4')
        run_import('lex_optimize4')
        for extension in ['py', 'pyc']:
            try:
                os.remove('opt4tab.{0}'.format(extension))
            except OSError:
                pass

    def test_lex_opt_alias(self):
        if False:
            for i in range(10):
                print('nop')
        try:
            os.remove('aliastab.py')
        except OSError:
            pass
        try:
            os.remove('aliastab.pyc')
        except OSError:
            pass
        try:
            os.remove('aliastab.pyo')
        except OSError:
            pass
        run_import('lex_opt_alias')
        result = sys.stdout.getvalue()
        self.assert_(check_expected(result, "(NUMBER,3,1,0)\n(+,'+',1,1)\n(NUMBER,4,1,2)\n"))
        self.assert_(os.path.exists('aliastab.py'))
        p = subprocess.Popen([sys.executable, '-O', 'lex_opt_alias.py'], stdout=subprocess.PIPE)
        result = p.stdout.read()
        self.assert_(check_expected(result, "(NUMBER,3,1,0)\n(+,'+',1,1)\n(NUMBER,4,1,2)\n"))
        if test_pyo:
            self.assert_(pymodule_out_exists('aliastab.pyo', 1))
            pymodule_out_remove('aliastab.pyo', 1)
        p = subprocess.Popen([sys.executable, '-OO', 'lex_opt_alias.py'], stdout=subprocess.PIPE)
        result = p.stdout.read()
        self.assert_(check_expected(result, "(NUMBER,3,1,0)\n(+,'+',1,1)\n(NUMBER,4,1,2)\n"))
        if test_pyo:
            self.assert_(pymodule_out_exists('aliastab.pyo', 2))
        try:
            os.remove('aliastab.py')
        except OSError:
            pass
        try:
            pymodule_out_remove('aliastab.pyc')
        except OSError:
            pass
        try:
            pymodule_out_remove('aliastab.pyo', 2)
        except OSError:
            pass

    def test_lex_many_tokens(self):
        if False:
            for i in range(10):
                print('nop')
        try:
            os.remove('manytab.py')
        except OSError:
            pass
        try:
            os.remove('manytab.pyc')
        except OSError:
            pass
        try:
            os.remove('manytab.pyo')
        except OSError:
            pass
        run_import('lex_many_tokens')
        result = sys.stdout.getvalue()
        self.assert_(check_expected(result, "(TOK34,'TOK34:',1,0)\n(TOK143,'TOK143:',1,7)\n(TOK269,'TOK269:',1,15)\n(TOK372,'TOK372:',1,23)\n(TOK452,'TOK452:',1,31)\n(TOK561,'TOK561:',1,39)\n(TOK999,'TOK999:',1,47)\n"))
        self.assert_(os.path.exists('manytab.py'))
        if implementation() == 'CPython':
            p = subprocess.Popen([sys.executable, '-O', 'lex_many_tokens.py'], stdout=subprocess.PIPE)
            result = p.stdout.read()
            self.assert_(check_expected(result, "(TOK34,'TOK34:',1,0)\n(TOK143,'TOK143:',1,7)\n(TOK269,'TOK269:',1,15)\n(TOK372,'TOK372:',1,23)\n(TOK452,'TOK452:',1,31)\n(TOK561,'TOK561:',1,39)\n(TOK999,'TOK999:',1,47)\n"))
            self.assert_(pymodule_out_exists('manytab.pyo', 1))
            pymodule_out_remove('manytab.pyo', 1)
        try:
            os.remove('manytab.py')
        except OSError:
            pass
        try:
            os.remove('manytab.pyc')
        except OSError:
            pass
        try:
            os.remove('manytab.pyo')
        except OSError:
            pass

class LexRunTests(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        sys.stderr = StringIO.StringIO()
        sys.stdout = StringIO.StringIO()

    def tearDown(self):
        if False:
            print('Hello World!')
        sys.stderr = sys.__stderr__
        sys.stdout = sys.__stdout__

    def test_lex_hedit(self):
        if False:
            while True:
                i = 10
        run_import('lex_hedit')
        result = sys.stdout.getvalue()
        self.assert_(check_expected(result, "(H_EDIT_DESCRIPTOR,'abc',1,0)\n(H_EDIT_DESCRIPTOR,'abcdefghij',1,6)\n(H_EDIT_DESCRIPTOR,'xy',1,20)\n"))

    def test_lex_state_try(self):
        if False:
            while True:
                i = 10
        run_import('lex_state_try')
        result = sys.stdout.getvalue()
        self.assert_(check_expected(result, "(NUMBER,'3',1,0)\n(PLUS,'+',1,2)\n(NUMBER,'4',1,4)\nEntering comment state\ncomment body LexToken(body_part,'This is a comment */',1,9)\n(PLUS,'+',1,30)\n(NUMBER,'10',1,32)\n"))
unittest.main()