"""Usage: unparse.py <path to source file>"""
import ast
import sys
from contextlib import contextmanager
from io import StringIO

@contextmanager
def nullcontext():
    if False:
        while True:
            i = 10
    yield
INFSTR = '1e' + repr(sys.float_info.max_10_exp + 1)

class _Precedence:
    """Precedence table that originated from python grammar."""
    TUPLE = 0
    YIELD = 1
    TEST = 2
    OR = 3
    AND = 4
    NOT = 5
    CMP = 6
    EXPR = 7
    BOR = EXPR
    BXOR = 8
    BAND = 9
    SHIFT = 10
    ARITH = 11
    TERM = 12
    FACTOR = 13
    POWER = 14
    AWAIT = 15
    ATOM = 16

def pnext(precedence):
    if False:
        return 10
    return min(precedence + 1, _Precedence.ATOM)

def interleave(inter, f, seq):
    if False:
        return 10
    'Call f on each item in seq, calling inter() in between.'
    seq = iter(seq)
    try:
        f(next(seq))
    except StopIteration:
        pass
    else:
        for x in seq:
            inter()
            f(x)
_SINGLE_QUOTES = ("'", '"')
_MULTI_QUOTES = ('"""', "'''")
_ALL_QUOTES = _SINGLE_QUOTES + _MULTI_QUOTES

def is_simple_tuple(slice_value):
    if False:
        print('Hello World!')
    return isinstance(slice_value, ast.Tuple) and slice_value.elts and (not any((isinstance(elt, ast.Starred) for elt in slice_value.elts)))

class Unparser:
    """Methods in this class recursively traverse an AST and
    output source code for the abstract syntax; original formatting
    is disregarded."""

    def __init__(self, py_ver_consistent=False, _avoid_backslashes=False):
        if False:
            while True:
                i = 10
        'Traverse an AST and generate its source.\n\n        Arguments:\n            py_ver_consistent (bool): if True, generate unparsed code that is\n                consistent between Python versions 3.5-3.11.\n\n        For legacy reasons, consistency is achieved by unparsing Python3 unicode literals\n        the way Python 2 would. This preserved Spack package hash consistency during the\n        python2/3 transition\n        '
        self.future_imports = []
        self._indent = 0
        self._py_ver_consistent = py_ver_consistent
        self._precedences = {}
        self._avoid_backslashes = _avoid_backslashes

    def items_view(self, traverser, items):
        if False:
            return 10
        'Traverse and separate the given *items* with a comma and append it to\n        the buffer. If *items* is a single item sequence, a trailing comma\n        will be added.'
        if len(items) == 1:
            traverser(items[0])
            self.write(',')
        else:
            interleave(lambda : self.write(', '), traverser, items)

    def visit(self, tree, output_file):
        if False:
            i = 10
            return i + 15
        'Traverse tree and write source code to output_file.'
        self.f = output_file
        self.dispatch(tree)
        self.f.flush()

    def fill(self, text=''):
        if False:
            return 10
        'Indent a piece of text, according to the current indentation level'
        self.f.write('\n' + '    ' * self._indent + text)

    def write(self, text):
        if False:
            print('Hello World!')
        'Append a piece of text to the current line.'
        self.f.write(str(text))

    class _Block:
        """A context manager for preparing the source for blocks. It adds
        the character ':', increases the indentation on enter and decreases
        the indentation on exit."""

        def __init__(self, unparser):
            if False:
                return 10
            self.unparser = unparser

        def __enter__(self):
            if False:
                for i in range(10):
                    print('nop')
            self.unparser.write(':')
            self.unparser._indent += 1

        def __exit__(self, exc_type, exc_value, traceback):
            if False:
                return 10
            self.unparser._indent -= 1

    def block(self):
        if False:
            while True:
                i = 10
        return self._Block(self)

    @contextmanager
    def delimit(self, start, end):
        if False:
            for i in range(10):
                print('nop')
        'A context manager for preparing the source for expressions. It adds\n        *start* to the buffer and enters, after exit it adds *end*.'
        self.write(start)
        yield
        self.write(end)

    def delimit_if(self, start, end, condition):
        if False:
            return 10
        if condition:
            return self.delimit(start, end)
        else:
            return nullcontext()

    def require_parens(self, precedence, node):
        if False:
            return 10
        'Shortcut to adding precedence related parens'
        return self.delimit_if('(', ')', self.get_precedence(node) > precedence)

    def get_precedence(self, node):
        if False:
            for i in range(10):
                print('nop')
        return self._precedences.get(node, _Precedence.TEST)

    def set_precedence(self, precedence, *nodes):
        if False:
            for i in range(10):
                print('nop')
        for node in nodes:
            self._precedences[node] = precedence

    def dispatch(self, tree):
        if False:
            return 10
        'Dispatcher function, dispatching tree type T to method _T.'
        if isinstance(tree, list):
            for node in tree:
                self.dispatch(node)
            return
        meth = getattr(self, 'visit_' + tree.__class__.__name__)
        meth(tree)

    def visit_Module(self, tree):
        if False:
            print('Hello World!')
        for stmt in tree.body:
            self.dispatch(stmt)

    def visit_Interactive(self, tree):
        if False:
            print('Hello World!')
        for stmt in tree.body:
            self.dispatch(stmt)

    def visit_Expression(self, tree):
        if False:
            return 10
        self.dispatch(tree.body)

    def visit_Expr(self, tree):
        if False:
            i = 10
            return i + 15
        self.fill()
        self.set_precedence(_Precedence.YIELD, tree.value)
        self.dispatch(tree.value)

    def visit_NamedExpr(self, tree):
        if False:
            return 10
        with self.require_parens(_Precedence.TUPLE, tree):
            self.set_precedence(_Precedence.ATOM, tree.target, tree.value)
            self.dispatch(tree.target)
            self.write(' := ')
            self.dispatch(tree.value)

    def visit_Import(self, node):
        if False:
            print('Hello World!')
        self.fill('import ')
        interleave(lambda : self.write(', '), self.dispatch, node.names)

    def visit_ImportFrom(self, node):
        if False:
            print('Hello World!')
        if node.module and node.module == '__future__':
            self.future_imports.extend((n.name for n in node.names))
        self.fill('from ')
        self.write('.' * node.level)
        if node.module:
            self.write(node.module)
        self.write(' import ')
        interleave(lambda : self.write(', '), self.dispatch, node.names)

    def visit_Assign(self, node):
        if False:
            return 10
        self.fill()
        for target in node.targets:
            self.dispatch(target)
            self.write(' = ')
        self.dispatch(node.value)

    def visit_AugAssign(self, node):
        if False:
            while True:
                i = 10
        self.fill()
        self.dispatch(node.target)
        self.write(' ' + self.binop[node.op.__class__.__name__] + '= ')
        self.dispatch(node.value)

    def visit_AnnAssign(self, node):
        if False:
            while True:
                i = 10
        self.fill()
        with self.delimit_if('(', ')', not node.simple and isinstance(node.target, ast.Name)):
            self.dispatch(node.target)
        self.write(': ')
        self.dispatch(node.annotation)
        if node.value:
            self.write(' = ')
            self.dispatch(node.value)

    def visit_Return(self, node):
        if False:
            print('Hello World!')
        self.fill('return')
        if node.value:
            self.write(' ')
            self.dispatch(node.value)

    def visit_Pass(self, node):
        if False:
            while True:
                i = 10
        self.fill('pass')

    def visit_Break(self, node):
        if False:
            print('Hello World!')
        self.fill('break')

    def visit_Continue(self, node):
        if False:
            for i in range(10):
                print('nop')
        self.fill('continue')

    def visit_Delete(self, node):
        if False:
            while True:
                i = 10
        self.fill('del ')
        interleave(lambda : self.write(', '), self.dispatch, node.targets)

    def visit_Assert(self, node):
        if False:
            for i in range(10):
                print('nop')
        self.fill('assert ')
        self.dispatch(node.test)
        if node.msg:
            self.write(', ')
            self.dispatch(node.msg)

    def visit_Global(self, node):
        if False:
            for i in range(10):
                print('nop')
        self.fill('global ')
        interleave(lambda : self.write(', '), self.write, node.names)

    def visit_Nonlocal(self, node):
        if False:
            print('Hello World!')
        self.fill('nonlocal ')
        interleave(lambda : self.write(', '), self.write, node.names)

    def visit_Await(self, node):
        if False:
            print('Hello World!')
        with self.require_parens(_Precedence.AWAIT, node):
            self.write('await')
            if node.value:
                self.write(' ')
                self.set_precedence(_Precedence.ATOM, node.value)
                self.dispatch(node.value)

    def visit_Yield(self, node):
        if False:
            print('Hello World!')
        with self.require_parens(_Precedence.YIELD, node):
            self.write('yield')
            if node.value:
                self.write(' ')
                self.set_precedence(_Precedence.ATOM, node.value)
                self.dispatch(node.value)

    def visit_YieldFrom(self, node):
        if False:
            i = 10
            return i + 15
        with self.require_parens(_Precedence.YIELD, node):
            self.write('yield from')
            if node.value:
                self.write(' ')
                self.set_precedence(_Precedence.ATOM, node.value)
                self.dispatch(node.value)

    def visit_Raise(self, node):
        if False:
            i = 10
            return i + 15
        self.fill('raise')
        if not node.exc:
            assert not node.cause
            return
        self.write(' ')
        self.dispatch(node.exc)
        if node.cause:
            self.write(' from ')
            self.dispatch(node.cause)

    def visit_Try(self, node):
        if False:
            for i in range(10):
                print('nop')
        self.fill('try')
        with self.block():
            self.dispatch(node.body)
        for ex in node.handlers:
            self.dispatch(ex)
        if node.orelse:
            self.fill('else')
            with self.block():
                self.dispatch(node.orelse)
        if node.finalbody:
            self.fill('finally')
            with self.block():
                self.dispatch(node.finalbody)

    def visit_ExceptHandler(self, node):
        if False:
            return 10
        self.fill('except')
        if node.type:
            self.write(' ')
            self.dispatch(node.type)
        if node.name:
            self.write(' as ')
            self.write(node.name)
        with self.block():
            self.dispatch(node.body)

    def visit_ClassDef(self, node):
        if False:
            while True:
                i = 10
        self.write('\n')
        for deco in node.decorator_list:
            self.fill('@')
            self.dispatch(deco)
        self.fill('class ' + node.name)
        if getattr(node, 'type_params', False):
            self.write('[')
            interleave(lambda : self.write(', '), self.dispatch, node.type_params)
            self.write(']')
        with self.delimit_if('(', ')', condition=node.bases or node.keywords):
            comma = False
            for e in node.bases:
                if comma:
                    self.write(', ')
                else:
                    comma = True
                self.dispatch(e)
            for e in node.keywords:
                if comma:
                    self.write(', ')
                else:
                    comma = True
                self.dispatch(e)
        with self.block():
            self.dispatch(node.body)

    def visit_FunctionDef(self, node):
        if False:
            print('Hello World!')
        self.__FunctionDef_helper(node, 'def')

    def visit_AsyncFunctionDef(self, node):
        if False:
            i = 10
            return i + 15
        self.__FunctionDef_helper(node, 'async def')

    def __FunctionDef_helper(self, node, fill_suffix):
        if False:
            while True:
                i = 10
        self.write('\n')
        for deco in node.decorator_list:
            self.fill('@')
            self.dispatch(deco)
        def_str = fill_suffix + ' ' + node.name
        self.fill(def_str)
        if getattr(node, 'type_params', False):
            self.write('[')
            interleave(lambda : self.write(', '), self.dispatch, node.type_params)
            self.write(']')
        with self.delimit('(', ')'):
            self.dispatch(node.args)
        if getattr(node, 'returns', False):
            self.write(' -> ')
            self.dispatch(node.returns)
        with self.block():
            self.dispatch(node.body)

    def visit_For(self, node):
        if False:
            for i in range(10):
                print('nop')
        self.__For_helper('for ', node)

    def visit_AsyncFor(self, node):
        if False:
            while True:
                i = 10
        self.__For_helper('async for ', node)

    def __For_helper(self, fill, node):
        if False:
            print('Hello World!')
        self.fill(fill)
        self.dispatch(node.target)
        self.write(' in ')
        self.dispatch(node.iter)
        with self.block():
            self.dispatch(node.body)
        if node.orelse:
            self.fill('else')
            with self.block():
                self.dispatch(node.orelse)

    def visit_If(self, node):
        if False:
            i = 10
            return i + 15
        self.fill('if ')
        self.dispatch(node.test)
        with self.block():
            self.dispatch(node.body)
        while node.orelse and len(node.orelse) == 1 and isinstance(node.orelse[0], ast.If):
            node = node.orelse[0]
            self.fill('elif ')
            self.dispatch(node.test)
            with self.block():
                self.dispatch(node.body)
        if node.orelse:
            self.fill('else')
            with self.block():
                self.dispatch(node.orelse)

    def visit_While(self, node):
        if False:
            print('Hello World!')
        self.fill('while ')
        self.dispatch(node.test)
        with self.block():
            self.dispatch(node.body)
        if node.orelse:
            self.fill('else')
            with self.block():
                self.dispatch(node.orelse)

    def _generic_With(self, node, async_=False):
        if False:
            i = 10
            return i + 15
        self.fill('async with ' if async_ else 'with ')
        if hasattr(node, 'items'):
            interleave(lambda : self.write(', '), self.dispatch, node.items)
        else:
            self.dispatch(node.context_expr)
            if node.optional_vars:
                self.write(' as ')
                self.dispatch(node.optional_vars)
        with self.block():
            self.dispatch(node.body)

    def visit_With(self, node):
        if False:
            while True:
                i = 10
        self._generic_With(node)

    def visit_AsyncWith(self, node):
        if False:
            for i in range(10):
                print('nop')
        self._generic_With(node, async_=True)

    def _str_literal_helper(self, string, quote_types=_ALL_QUOTES, escape_special_whitespace=False):
        if False:
            i = 10
            return i + 15
        'Helper for writing string literals, minimizing escapes.\n        Returns the tuple (string literal to write, possible quote types).\n        '

        def escape_char(c):
            if False:
                while True:
                    i = 10
            if not escape_special_whitespace and c in '\n\t':
                return c
            if c == '\\' or not c.isprintable():
                return c.encode('unicode_escape').decode('ascii')
            return c
        escaped_string = ''.join(map(escape_char, string))
        possible_quotes = quote_types
        if '\n' in escaped_string:
            possible_quotes = [q for q in possible_quotes if q in _MULTI_QUOTES]
        possible_quotes = [q for q in possible_quotes if q not in escaped_string]
        if not possible_quotes:
            string = repr(string)
            quote = next((q for q in quote_types if string[0] in q), string[0])
            return (string[1:-1], [quote])
        if escaped_string:
            possible_quotes.sort(key=lambda q: q[0] == escaped_string[-1])
            if possible_quotes[0][0] == escaped_string[-1]:
                assert len(possible_quotes[0]) == 3
                escaped_string = escaped_string[:-1] + '\\' + escaped_string[-1]
        return (escaped_string, possible_quotes)

    def _write_str_avoiding_backslashes(self, string, quote_types=_ALL_QUOTES):
        if False:
            return 10
        'Write string literal value w/a best effort attempt to avoid backslashes.'
        (string, quote_types) = self._str_literal_helper(string, quote_types=quote_types)
        quote_type = quote_types[0]
        self.write('{quote_type}{string}{quote_type}'.format(quote_type=quote_type, string=string))

    def visit_Bytes(self, node):
        if False:
            while True:
                i = 10
        self.write(repr(node.s))

    def visit_Str(self, tree):
        if False:
            while True:
                i = 10
        if self._py_ver_consistent and repr(tree.s).startswith("'\\u"):
            self.write('u')
        self._write_constant(tree.s)

    def visit_JoinedStr(self, node):
        if False:
            return 10
        self.write('f')
        if self._avoid_backslashes:
            string = StringIO()
            self._fstring_JoinedStr(node, string.write)
            self._write_str_avoiding_backslashes(string.getvalue())
            return
        buffer = []
        for value in node.values:
            meth = getattr(self, '_fstring_' + type(value).__name__)
            string = StringIO()
            meth(value, string.write)
            buffer.append((string.getvalue(), isinstance(value, ast.Constant)))
        new_buffer = []
        quote_types = _ALL_QUOTES
        for (value, is_constant) in buffer:
            (value, quote_types) = self._str_literal_helper(value, quote_types=quote_types, escape_special_whitespace=is_constant)
            new_buffer.append(value)
        value = ''.join(new_buffer)
        quote_type = quote_types[0]
        self.write('{quote_type}{value}{quote_type}'.format(quote_type=quote_type, value=value))

    def visit_FormattedValue(self, node):
        if False:
            print('Hello World!')
        self.write('f')
        string = StringIO()
        self._fstring_JoinedStr(node, string.write)
        self._write_str_avoiding_backslashes(string.getvalue())

    def _fstring_JoinedStr(self, node, write):
        if False:
            return 10
        for value in node.values:
            print('   ', value)
            meth = getattr(self, '_fstring_' + type(value).__name__)
            print(meth)
            meth(value, write)

    def _fstring_Str(self, node, write):
        if False:
            print('Hello World!')
        value = node.s.replace('{', '{{').replace('}', '}}')
        write(value)

    def _fstring_Constant(self, node, write):
        if False:
            print('Hello World!')
        assert isinstance(node.value, str)
        value = node.value.replace('{', '{{').replace('}', '}}')
        write(value)

    def _fstring_FormattedValue(self, node, write):
        if False:
            while True:
                i = 10
        write('{')
        expr = StringIO()
        unparser = type(self)(py_ver_consistent=self._py_ver_consistent, _avoid_backslashes=True)
        unparser.set_precedence(pnext(_Precedence.TEST), node.value)
        unparser.visit(node.value, expr)
        expr = expr.getvalue().rstrip('\n')
        if expr.startswith('{'):
            write(' ')
        if '\\' in expr:
            raise ValueError('Unable to avoid backslash in f-string expression part')
        write(expr)
        if node.conversion != -1:
            conversion = chr(node.conversion)
            assert conversion in 'sra'
            write('!{conversion}'.format(conversion=conversion))
        if node.format_spec:
            write(':')
            meth = getattr(self, '_fstring_' + type(node.format_spec).__name__)
            meth(node.format_spec, write)
        write('}')

    def visit_Name(self, node):
        if False:
            print('Hello World!')
        self.write(node.id)

    def visit_NameConstant(self, node):
        if False:
            print('Hello World!')
        self.write(repr(node.value))

    def _write_constant(self, value):
        if False:
            return 10
        if isinstance(value, (float, complex)):
            self.write(repr(value).replace('inf', INFSTR))
        elif isinstance(value, str) and self._py_ver_consistent:
            raw = repr(value.encode('raw_unicode_escape')).lstrip('b')
            if raw.startswith("'\\\\u"):
                raw = "'\\" + raw[3:]
            self.write(raw)
        elif self._avoid_backslashes and isinstance(value, str):
            self._write_str_avoiding_backslashes(value)
        else:
            self.write(repr(value))

    def visit_Constant(self, node):
        if False:
            while True:
                i = 10
        value = node.value
        if isinstance(value, tuple):
            with self.delimit('(', ')'):
                self.items_view(self._write_constant, value)
        elif value is Ellipsis:
            self.write('...')
        else:
            if node.kind == 'u':
                self.write('u')
            self._write_constant(node.value)

    def visit_Num(self, node):
        if False:
            print('Hello World!')
        repr_n = repr(node.n)
        self.write(repr_n.replace('inf', INFSTR))

    def visit_List(self, node):
        if False:
            for i in range(10):
                print('nop')
        with self.delimit('[', ']'):
            interleave(lambda : self.write(', '), self.dispatch, node.elts)

    def visit_ListComp(self, node):
        if False:
            while True:
                i = 10
        with self.delimit('[', ']'):
            self.dispatch(node.elt)
            for gen in node.generators:
                self.dispatch(gen)

    def visit_GeneratorExp(self, node):
        if False:
            print('Hello World!')
        with self.delimit('(', ')'):
            self.dispatch(node.elt)
            for gen in node.generators:
                self.dispatch(gen)

    def visit_SetComp(self, node):
        if False:
            i = 10
            return i + 15
        with self.delimit('{', '}'):
            self.dispatch(node.elt)
            for gen in node.generators:
                self.dispatch(gen)

    def visit_DictComp(self, node):
        if False:
            return 10
        with self.delimit('{', '}'):
            self.dispatch(node.key)
            self.write(': ')
            self.dispatch(node.value)
            for gen in node.generators:
                self.dispatch(gen)

    def visit_comprehension(self, node):
        if False:
            print('Hello World!')
        if getattr(node, 'is_async', False):
            self.write(' async for ')
        else:
            self.write(' for ')
        self.set_precedence(_Precedence.TUPLE, node.target)
        self.dispatch(node.target)
        self.write(' in ')
        self.set_precedence(pnext(_Precedence.TEST), node.iter, *node.ifs)
        self.dispatch(node.iter)
        for if_clause in node.ifs:
            self.write(' if ')
            self.dispatch(if_clause)

    def visit_IfExp(self, node):
        if False:
            print('Hello World!')
        with self.require_parens(_Precedence.TEST, node):
            self.set_precedence(pnext(_Precedence.TEST), node.body, node.test)
            self.dispatch(node.body)
            self.write(' if ')
            self.dispatch(node.test)
            self.write(' else ')
            self.set_precedence(_Precedence.TEST, node.orelse)
            self.dispatch(node.orelse)

    def visit_Set(self, node):
        if False:
            for i in range(10):
                print('nop')
        assert node.elts
        with self.delimit('{', '}'):
            interleave(lambda : self.write(', '), self.dispatch, node.elts)

    def visit_Dict(self, node):
        if False:
            while True:
                i = 10

        def write_key_value_pair(k, v):
            if False:
                print('Hello World!')
            self.dispatch(k)
            self.write(': ')
            self.dispatch(v)

        def write_item(item):
            if False:
                print('Hello World!')
            (k, v) = item
            if k is None:
                self.write('**')
                self.set_precedence(_Precedence.EXPR, v)
                self.dispatch(v)
            else:
                write_key_value_pair(k, v)
        with self.delimit('{', '}'):
            interleave(lambda : self.write(', '), write_item, zip(node.keys, node.values))

    def visit_Tuple(self, node):
        if False:
            print('Hello World!')
        with self.delimit('(', ')'):
            self.items_view(self.dispatch, node.elts)
    unop = {'Invert': '~', 'Not': 'not', 'UAdd': '+', 'USub': '-'}
    unop_precedence = {'~': _Precedence.FACTOR, 'not': _Precedence.NOT, '+': _Precedence.FACTOR, '-': _Precedence.FACTOR}

    def visit_UnaryOp(self, node):
        if False:
            return 10
        operator = self.unop[node.op.__class__.__name__]
        operator_precedence = self.unop_precedence[operator]
        with self.require_parens(operator_precedence, node):
            self.write(operator)
            if operator_precedence != _Precedence.FACTOR:
                self.write(' ')
            self.set_precedence(operator_precedence, node.operand)
            self.dispatch(node.operand)
    binop = {'Add': '+', 'Sub': '-', 'Mult': '*', 'MatMult': '@', 'Div': '/', 'Mod': '%', 'LShift': '<<', 'RShift': '>>', 'BitOr': '|', 'BitXor': '^', 'BitAnd': '&', 'FloorDiv': '//', 'Pow': '**'}
    binop_precedence = {'+': _Precedence.ARITH, '-': _Precedence.ARITH, '*': _Precedence.TERM, '@': _Precedence.TERM, '/': _Precedence.TERM, '%': _Precedence.TERM, '<<': _Precedence.SHIFT, '>>': _Precedence.SHIFT, '|': _Precedence.BOR, '^': _Precedence.BXOR, '&': _Precedence.BAND, '//': _Precedence.TERM, '**': _Precedence.POWER}
    binop_rassoc = frozenset(('**',))

    def visit_BinOp(self, node):
        if False:
            for i in range(10):
                print('nop')
        operator = self.binop[node.op.__class__.__name__]
        operator_precedence = self.binop_precedence[operator]
        with self.require_parens(operator_precedence, node):
            if operator in self.binop_rassoc:
                left_precedence = pnext(operator_precedence)
                right_precedence = operator_precedence
            else:
                left_precedence = operator_precedence
                right_precedence = pnext(operator_precedence)
            self.set_precedence(left_precedence, node.left)
            self.dispatch(node.left)
            self.write(' %s ' % operator)
            self.set_precedence(right_precedence, node.right)
            self.dispatch(node.right)
    cmpops = {'Eq': '==', 'NotEq': '!=', 'Lt': '<', 'LtE': '<=', 'Gt': '>', 'GtE': '>=', 'Is': 'is', 'IsNot': 'is not', 'In': 'in', 'NotIn': 'not in'}

    def visit_Compare(self, node):
        if False:
            for i in range(10):
                print('nop')
        with self.require_parens(_Precedence.CMP, node):
            self.set_precedence(pnext(_Precedence.CMP), node.left, *node.comparators)
            self.dispatch(node.left)
            for (o, e) in zip(node.ops, node.comparators):
                self.write(' ' + self.cmpops[o.__class__.__name__] + ' ')
                self.dispatch(e)
    boolops = {'And': 'and', 'Or': 'or'}
    boolop_precedence = {'and': _Precedence.AND, 'or': _Precedence.OR}

    def visit_BoolOp(self, node):
        if False:
            print('Hello World!')
        operator = self.boolops[node.op.__class__.__name__]
        op = {'precedence': self.boolop_precedence[operator]}

        def increasing_level_dispatch(node):
            if False:
                i = 10
                return i + 15
            op['precedence'] = pnext(op['precedence'])
            self.set_precedence(op['precedence'], node)
            self.dispatch(node)
        with self.require_parens(op['precedence'], node):
            s = ' %s ' % operator
            interleave(lambda : self.write(s), increasing_level_dispatch, node.values)

    def visit_Attribute(self, node):
        if False:
            return 10
        self.set_precedence(_Precedence.ATOM, node.value)
        self.dispatch(node.value)
        num_type = getattr(ast, 'Constant', getattr(ast, 'Num', None))
        if isinstance(node.value, num_type) and isinstance(node.value.n, int):
            self.write(' ')
        self.write('.')
        self.write(node.attr)

    def visit_Call(self, node):
        if False:
            for i in range(10):
                print('nop')
        self.set_precedence(_Precedence.ATOM, node.func)
        args = node.args
        self.dispatch(node.func)
        with self.delimit('(', ')'):
            comma = False
            for e in args:
                if comma:
                    self.write(', ')
                else:
                    comma = True
                self.dispatch(e)
            for e in node.keywords:
                if comma:
                    self.write(', ')
                else:
                    comma = True
                self.dispatch(e)

    def visit_Subscript(self, node):
        if False:
            while True:
                i = 10
        self.set_precedence(_Precedence.ATOM, node.value)
        self.dispatch(node.value)
        with self.delimit('[', ']'):
            if is_simple_tuple(node.slice):
                self.items_view(self.dispatch, node.slice.elts)
            else:
                self.dispatch(node.slice)

    def visit_Starred(self, node):
        if False:
            return 10
        self.write('*')
        self.set_precedence(_Precedence.EXPR, node.value)
        self.dispatch(node.value)

    def visit_Ellipsis(self, node):
        if False:
            return 10
        self.write('...')

    def visit_Index(self, node):
        if False:
            while True:
                i = 10
        if is_simple_tuple(node.value):
            self.set_precedence(_Precedence.ATOM, node.value)
            self.items_view(self.dispatch, node.value.elts)
        else:
            self.set_precedence(_Precedence.TUPLE, node.value)
            self.dispatch(node.value)

    def visit_Slice(self, node):
        if False:
            print('Hello World!')
        if node.lower:
            self.dispatch(node.lower)
        self.write(':')
        if node.upper:
            self.dispatch(node.upper)
        if node.step:
            self.write(':')
            self.dispatch(node.step)

    def visit_ExtSlice(self, node):
        if False:
            i = 10
            return i + 15
        interleave(lambda : self.write(', '), self.dispatch, node.dims)

    def visit_arg(self, node):
        if False:
            while True:
                i = 10
        self.write(node.arg)
        if node.annotation:
            self.write(': ')
            self.dispatch(node.annotation)

    def visit_arguments(self, node):
        if False:
            i = 10
            return i + 15
        first = True
        all_args = getattr(node, 'posonlyargs', []) + node.args
        defaults = [None] * (len(all_args) - len(node.defaults)) + node.defaults
        for (index, elements) in enumerate(zip(all_args, defaults), 1):
            (a, d) = elements
            if first:
                first = False
            else:
                self.write(', ')
            self.dispatch(a)
            if d:
                self.write('=')
                self.dispatch(d)
            if index == len(getattr(node, 'posonlyargs', ())):
                self.write(', /')
        if node.vararg or getattr(node, 'kwonlyargs', False):
            if first:
                first = False
            else:
                self.write(', ')
            self.write('*')
            if node.vararg:
                self.write(node.vararg.arg)
                if node.vararg.annotation:
                    self.write(': ')
                    self.dispatch(node.vararg.annotation)
        if getattr(node, 'kwonlyargs', False):
            for (a, d) in zip(node.kwonlyargs, node.kw_defaults):
                if first:
                    first = False
                else:
                    self.write(', ')
                (self.dispatch(a),)
                if d:
                    self.write('=')
                    self.dispatch(d)
        if node.kwarg:
            if first:
                first = False
            else:
                self.write(', ')
            self.write('**' + node.kwarg.arg)
            if node.kwarg.annotation:
                self.write(': ')
                self.dispatch(node.kwarg.annotation)

    def visit_keyword(self, node):
        if False:
            return 10
        if node.arg is None:
            self.write('**')
        else:
            self.write(node.arg)
            self.write('=')
        self.dispatch(node.value)

    def visit_Lambda(self, node):
        if False:
            print('Hello World!')
        with self.require_parens(_Precedence.TEST, node):
            self.write('lambda ')
            self.dispatch(node.args)
            self.write(': ')
            self.set_precedence(_Precedence.TEST, node.body)
            self.dispatch(node.body)

    def visit_alias(self, node):
        if False:
            for i in range(10):
                print('nop')
        self.write(node.name)
        if node.asname:
            self.write(' as ' + node.asname)

    def visit_withitem(self, node):
        if False:
            i = 10
            return i + 15
        self.dispatch(node.context_expr)
        if node.optional_vars:
            self.write(' as ')
            self.dispatch(node.optional_vars)

    def visit_Match(self, node):
        if False:
            for i in range(10):
                print('nop')
        self.fill('match ')
        self.dispatch(node.subject)
        with self.block():
            for case in node.cases:
                self.dispatch(case)

    def visit_match_case(self, node):
        if False:
            i = 10
            return i + 15
        self.fill('case ')
        self.dispatch(node.pattern)
        if node.guard:
            self.write(' if ')
            self.dispatch(node.guard)
        with self.block():
            self.dispatch(node.body)

    def visit_MatchValue(self, node):
        if False:
            i = 10
            return i + 15
        self.dispatch(node.value)

    def visit_MatchSingleton(self, node):
        if False:
            i = 10
            return i + 15
        self._write_constant(node.value)

    def visit_MatchSequence(self, node):
        if False:
            print('Hello World!')
        with self.delimit('[', ']'):
            interleave(lambda : self.write(', '), self.dispatch, node.patterns)

    def visit_MatchStar(self, node):
        if False:
            i = 10
            return i + 15
        name = node.name
        if name is None:
            name = '_'
        self.write('*{}'.format(name))

    def visit_MatchMapping(self, node):
        if False:
            while True:
                i = 10

        def write_key_pattern_pair(pair):
            if False:
                i = 10
                return i + 15
            (k, p) = pair
            self.dispatch(k)
            self.write(': ')
            self.dispatch(p)
        with self.delimit('{', '}'):
            keys = node.keys
            interleave(lambda : self.write(', '), write_key_pattern_pair, zip(keys, node.patterns))
            rest = node.rest
            if rest is not None:
                if keys:
                    self.write(', ')
                self.write('**{}'.format(rest))

    def visit_MatchClass(self, node):
        if False:
            print('Hello World!')
        self.set_precedence(_Precedence.ATOM, node.cls)
        self.dispatch(node.cls)
        with self.delimit('(', ')'):
            patterns = node.patterns
            interleave(lambda : self.write(', '), self.dispatch, patterns)
            attrs = node.kwd_attrs
            if attrs:

                def write_attr_pattern(pair):
                    if False:
                        while True:
                            i = 10
                    (attr, pattern) = pair
                    self.write('{}='.format(attr))
                    self.dispatch(pattern)
                if patterns:
                    self.write(', ')
                interleave(lambda : self.write(', '), write_attr_pattern, zip(attrs, node.kwd_patterns))

    def visit_MatchAs(self, node):
        if False:
            print('Hello World!')
        name = node.name
        pattern = node.pattern
        if name is None:
            self.write('_')
        elif pattern is None:
            self.write(node.name)
        else:
            with self.require_parens(_Precedence.TEST, node):
                self.set_precedence(_Precedence.BOR, node.pattern)
                self.dispatch(node.pattern)
                self.write(' as {}'.format(node.name))

    def visit_MatchOr(self, node):
        if False:
            print('Hello World!')
        with self.require_parens(_Precedence.BOR, node):
            self.set_precedence(pnext(_Precedence.BOR), *node.patterns)
            interleave(lambda : self.write(' | '), self.dispatch, node.patterns)

    def visit_TypeAlias(self, node):
        if False:
            return 10
        self.fill('type ')
        self.dispatch(node.name)
        if node.type_params:
            self.write('[')
            interleave(lambda : self.write(', '), self.dispatch, node.type_params)
            self.write(']')
        self.write(' = ')
        self.dispatch(node.value)

    def visit_TypeVar(self, node):
        if False:
            while True:
                i = 10
        self.write(node.name)
        if node.bound:
            self.write(': ')
            self.dispatch(node.bound)

    def visit_TypeVarTuple(self, node):
        if False:
            i = 10
            return i + 15
        self.write('*')
        self.write(node.name)

    def visit_ParamSpec(self, node):
        if False:
            while True:
                i = 10
        self.write('**')
        self.write(node.name)