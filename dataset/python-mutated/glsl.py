from __future__ import annotations
from sympy.core import Basic, S
from sympy.core.function import Lambda
from sympy.core.numbers import equal_valued
from sympy.printing.codeprinter import CodePrinter
from sympy.printing.precedence import precedence
from functools import reduce
known_functions = {'Abs': 'abs', 'sin': 'sin', 'cos': 'cos', 'tan': 'tan', 'acos': 'acos', 'asin': 'asin', 'atan': 'atan', 'atan2': 'atan', 'ceiling': 'ceil', 'floor': 'floor', 'sign': 'sign', 'exp': 'exp', 'log': 'log', 'add': 'add', 'sub': 'sub', 'mul': 'mul', 'pow': 'pow'}

class GLSLPrinter(CodePrinter):
    """
    Rudimentary, generic GLSL printing tools.

    Additional settings:
    'use_operators': Boolean (should the printer use operators for +,-,*, or functions?)
    """
    _not_supported: set[Basic] = set()
    printmethod = '_glsl'
    language = 'GLSL'
    _default_settings = {'use_operators': True, 'zero': 0, 'mat_nested': False, 'mat_separator': ',\n', 'mat_transpose': False, 'array_type': 'float', 'glsl_types': True, 'order': None, 'full_prec': 'auto', 'precision': 9, 'user_functions': {}, 'human': True, 'allow_unknown_functions': False, 'contract': True, 'error_on_reserved': False, 'reserved_word_suffix': '_'}

    def __init__(self, settings={}):
        if False:
            while True:
                i = 10
        CodePrinter.__init__(self, settings)
        self.known_functions = dict(known_functions)
        userfuncs = settings.get('user_functions', {})
        self.known_functions.update(userfuncs)

    def _rate_index_position(self, p):
        if False:
            for i in range(10):
                print('nop')
        return p * 5

    def _get_statement(self, codestring):
        if False:
            while True:
                i = 10
        return '%s;' % codestring

    def _get_comment(self, text):
        if False:
            for i in range(10):
                print('nop')
        return '// {}'.format(text)

    def _declare_number_const(self, name, value):
        if False:
            return 10
        return 'float {} = {};'.format(name, value)

    def _format_code(self, lines):
        if False:
            while True:
                i = 10
        return self.indent_code(lines)

    def indent_code(self, code):
        if False:
            return 10
        'Accepts a string of code or a list of code lines'
        if isinstance(code, str):
            code_lines = self.indent_code(code.splitlines(True))
            return ''.join(code_lines)
        tab = '   '
        inc_token = ('{', '(', '{\n', '(\n')
        dec_token = ('}', ')')
        code = [line.lstrip(' \t') for line in code]
        increase = [int(any(map(line.endswith, inc_token))) for line in code]
        decrease = [int(any(map(line.startswith, dec_token))) for line in code]
        pretty = []
        level = 0
        for (n, line) in enumerate(code):
            if line in ('', '\n'):
                pretty.append(line)
                continue
            level -= decrease[n]
            pretty.append('%s%s' % (tab * level, line))
            level += increase[n]
        return pretty

    def _print_MatrixBase(self, mat):
        if False:
            i = 10
            return i + 15
        mat_separator = self._settings['mat_separator']
        mat_transpose = self._settings['mat_transpose']
        column_vector = mat.rows == 1 if mat_transpose else mat.cols == 1
        A = mat.transpose() if mat_transpose != column_vector else mat
        glsl_types = self._settings['glsl_types']
        array_type = self._settings['array_type']
        array_size = A.cols * A.rows
        array_constructor = '{}[{}]'.format(array_type, array_size)
        if A.cols == 1:
            return self._print(A[0])
        if A.rows <= 4 and A.cols <= 4 and glsl_types:
            if A.rows == 1:
                return 'vec{}{}'.format(A.cols, A.table(self, rowstart='(', rowend=')'))
            elif A.rows == A.cols:
                return 'mat{}({})'.format(A.rows, A.table(self, rowsep=', ', rowstart='', rowend=''))
            else:
                return 'mat{}x{}({})'.format(A.cols, A.rows, A.table(self, rowsep=', ', rowstart='', rowend=''))
        elif S.One in A.shape:
            return '{}({})'.format(array_constructor, A.table(self, rowsep=mat_separator, rowstart='', rowend=''))
        elif not self._settings['mat_nested']:
            return '{}(\n{}\n) /* a {}x{} matrix */'.format(array_constructor, A.table(self, rowsep=mat_separator, rowstart='', rowend=''), A.rows, A.cols)
        elif self._settings['mat_nested']:
            return '{}[{}][{}](\n{}\n)'.format(array_type, A.rows, A.cols, A.table(self, rowsep=mat_separator, rowstart='float[](', rowend=')'))

    def _print_SparseRepMatrix(self, mat):
        if False:
            i = 10
            return i + 15
        return self._print_not_supported(mat)

    def _traverse_matrix_indices(self, mat):
        if False:
            return 10
        mat_transpose = self._settings['mat_transpose']
        if mat_transpose:
            (rows, cols) = mat.shape
        else:
            (cols, rows) = mat.shape
        return ((i, j) for i in range(cols) for j in range(rows))

    def _print_MatrixElement(self, expr):
        if False:
            while True:
                i = 10
        nest = self._settings['mat_nested']
        glsl_types = self._settings['glsl_types']
        mat_transpose = self._settings['mat_transpose']
        if mat_transpose:
            (cols, rows) = expr.parent.shape
            (i, j) = (expr.j, expr.i)
        else:
            (rows, cols) = expr.parent.shape
            (i, j) = (expr.i, expr.j)
        pnt = self._print(expr.parent)
        if glsl_types and (rows <= 4 and cols <= 4 or nest):
            return '{}[{}][{}]'.format(pnt, i, j)
        else:
            return '{}[{}]'.format(pnt, i + j * rows)

    def _print_list(self, expr):
        if False:
            print('Hello World!')
        l = ', '.join((self._print(item) for item in expr))
        glsl_types = self._settings['glsl_types']
        array_type = self._settings['array_type']
        array_size = len(expr)
        array_constructor = '{}[{}]'.format(array_type, array_size)
        if array_size <= 4 and glsl_types:
            return 'vec{}({})'.format(array_size, l)
        else:
            return '{}({})'.format(array_constructor, l)
    _print_tuple = _print_list
    _print_Tuple = _print_list

    def _get_loop_opening_ending(self, indices):
        if False:
            i = 10
            return i + 15
        open_lines = []
        close_lines = []
        loopstart = 'for (int %(varble)s=%(start)s; %(varble)s<%(end)s; %(varble)s++){'
        for i in indices:
            open_lines.append(loopstart % {'varble': self._print(i.label), 'start': self._print(i.lower), 'end': self._print(i.upper + 1)})
            close_lines.append('}')
        return (open_lines, close_lines)

    def _print_Function_with_args(self, func, func_args):
        if False:
            while True:
                i = 10
        if func in self.known_functions:
            cond_func = self.known_functions[func]
            func = None
            if isinstance(cond_func, str):
                func = cond_func
            else:
                for (cond, func) in cond_func:
                    if cond(func_args):
                        break
            if func is not None:
                try:
                    return func(*[self.parenthesize(item, 0) for item in func_args])
                except TypeError:
                    return '{}({})'.format(func, self.stringify(func_args, ', '))
        elif isinstance(func, Lambda):
            return self._print(func(*func_args))
        else:
            return self._print_not_supported(func)

    def _print_Piecewise(self, expr):
        if False:
            while True:
                i = 10
        from sympy.codegen.ast import Assignment
        if expr.args[-1].cond != True:
            raise ValueError('All Piecewise expressions must contain an (expr, True) statement to be used as a default condition. Without one, the generated expression may not evaluate to anything under some condition.')
        lines = []
        if expr.has(Assignment):
            for (i, (e, c)) in enumerate(expr.args):
                if i == 0:
                    lines.append('if (%s) {' % self._print(c))
                elif i == len(expr.args) - 1 and c == True:
                    lines.append('else {')
                else:
                    lines.append('else if (%s) {' % self._print(c))
                code0 = self._print(e)
                lines.append(code0)
                lines.append('}')
            return '\n'.join(lines)
        else:
            ecpairs = ['((%s) ? (\n%s\n)\n' % (self._print(c), self._print(e)) for (e, c) in expr.args[:-1]]
            last_line = ': (\n%s\n)' % self._print(expr.args[-1].expr)
            return ': '.join(ecpairs) + last_line + ' '.join([')' * len(ecpairs)])

    def _print_Idx(self, expr):
        if False:
            while True:
                i = 10
        return self._print(expr.label)

    def _print_Indexed(self, expr):
        if False:
            return 10
        dims = expr.shape
        elem = S.Zero
        offset = S.One
        for i in reversed(range(expr.rank)):
            elem += expr.indices[i] * offset
            offset *= dims[i]
        return '{}[{}]'.format(self._print(expr.base.label), self._print(elem))

    def _print_Pow(self, expr):
        if False:
            for i in range(10):
                print('nop')
        PREC = precedence(expr)
        if equal_valued(expr.exp, -1):
            return '1.0/%s' % self.parenthesize(expr.base, PREC)
        elif equal_valued(expr.exp, 0.5):
            return 'sqrt(%s)' % self._print(expr.base)
        else:
            try:
                e = self._print(float(expr.exp))
            except TypeError:
                e = self._print(expr.exp)
            return self._print_Function_with_args('pow', (self._print(expr.base), e))

    def _print_int(self, expr):
        if False:
            return 10
        return str(float(expr))

    def _print_Rational(self, expr):
        if False:
            return 10
        return '{}.0/{}.0'.format(expr.p, expr.q)

    def _print_Relational(self, expr):
        if False:
            while True:
                i = 10
        lhs_code = self._print(expr.lhs)
        rhs_code = self._print(expr.rhs)
        op = expr.rel_op
        return '{} {} {}'.format(lhs_code, op, rhs_code)

    def _print_Add(self, expr, order=None):
        if False:
            for i in range(10):
                print('nop')
        if self._settings['use_operators']:
            return CodePrinter._print_Add(self, expr, order=order)
        terms = expr.as_ordered_terms()

        def partition(p, l):
            if False:
                print('Hello World!')
            return reduce(lambda x, y: (x[0] + [y], x[1]) if p(y) else (x[0], x[1] + [y]), l, ([], []))

        def add(a, b):
            if False:
                for i in range(10):
                    print('nop')
            return self._print_Function_with_args('add', (a, b))
        (neg, pos) = partition(lambda arg: arg.could_extract_minus_sign(), terms)
        if pos:
            s = pos = reduce(lambda a, b: add(a, b), (self._print(t) for t in pos))
        else:
            s = pos = self._print(self._settings['zero'])
        if neg:
            neg = reduce(lambda a, b: add(a, b), (self._print(-n) for n in neg))
            s = self._print_Function_with_args('sub', (pos, neg))
        return s

    def _print_Mul(self, expr, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        if self._settings['use_operators']:
            return CodePrinter._print_Mul(self, expr, **kwargs)
        terms = expr.as_ordered_factors()

        def mul(a, b):
            if False:
                return 10
            return self._print_Function_with_args('mul', (a, b))
        s = reduce(lambda a, b: mul(a, b), (self._print(t) for t in terms))
        return s

def glsl_code(expr, assign_to=None, **settings):
    if False:
        while True:
            i = 10
    'Converts an expr to a string of GLSL code\n\n    Parameters\n    ==========\n\n    expr : Expr\n        A SymPy expression to be converted.\n    assign_to : optional\n        When given, the argument is used for naming the variable or variables\n        to which the expression is assigned. Can be a string, ``Symbol``,\n        ``MatrixSymbol`` or ``Indexed`` type object. In cases where ``expr``\n        would be printed as an array, a list of string or ``Symbol`` objects\n        can also be passed.\n\n        This is helpful in case of line-wrapping, or for expressions that\n        generate multi-line statements.  It can also be used to spread an array-like\n        expression into multiple assignments.\n    use_operators: bool, optional\n        If set to False, then *,/,+,- operators will be replaced with functions\n        mul, add, and sub, which must be implemented by the user, e.g. for\n        implementing non-standard rings or emulated quad/octal precision.\n        [default=True]\n    glsl_types: bool, optional\n        Set this argument to ``False`` in order to avoid using the ``vec`` and ``mat``\n        types.  The printer will instead use arrays (or nested arrays).\n        [default=True]\n    mat_nested: bool, optional\n        GLSL version 4.3 and above support nested arrays (arrays of arrays).  Set this to ``True``\n        to render matrices as nested arrays.\n        [default=False]\n    mat_separator: str, optional\n        By default, matrices are rendered with newlines using this separator,\n        making them easier to read, but less compact.  By removing the newline\n        this option can be used to make them more vertically compact.\n        [default=\',\n\']\n    mat_transpose: bool, optional\n        GLSL\'s matrix multiplication implementation assumes column-major indexing.\n        By default, this printer ignores that convention. Setting this option to\n        ``True`` transposes all matrix output.\n        [default=False]\n    array_type: str, optional\n        The GLSL array constructor type.\n        [default=\'float\']\n    precision : integer, optional\n        The precision for numbers such as pi [default=15].\n    user_functions : dict, optional\n        A dictionary where keys are ``FunctionClass`` instances and values are\n        their string representations. Alternatively, the dictionary value can\n        be a list of tuples i.e. [(argument_test, js_function_string)]. See\n        below for examples.\n    human : bool, optional\n        If True, the result is a single string that may contain some constant\n        declarations for the number symbols. If False, the same information is\n        returned in a tuple of (symbols_to_declare, not_supported_functions,\n        code_text). [default=True].\n    contract: bool, optional\n        If True, ``Indexed`` instances are assumed to obey tensor contraction\n        rules and the corresponding nested loops over indices are generated.\n        Setting contract=False will not generate loops, instead the user is\n        responsible to provide values for the indices in the code.\n        [default=True].\n\n    Examples\n    ========\n\n    >>> from sympy import glsl_code, symbols, Rational, sin, ceiling, Abs\n    >>> x, tau = symbols("x, tau")\n    >>> glsl_code((2*tau)**Rational(7, 2))\n    \'8*sqrt(2)*pow(tau, 3.5)\'\n    >>> glsl_code(sin(x), assign_to="float y")\n    \'float y = sin(x);\'\n\n    Various GLSL types are supported:\n    >>> from sympy import Matrix, glsl_code\n    >>> glsl_code(Matrix([1,2,3]))\n    \'vec3(1, 2, 3)\'\n\n    >>> glsl_code(Matrix([[1, 2],[3, 4]]))\n    \'mat2(1, 2, 3, 4)\'\n\n    Pass ``mat_transpose = True`` to switch to column-major indexing:\n    >>> glsl_code(Matrix([[1, 2],[3, 4]]), mat_transpose = True)\n    \'mat2(1, 3, 2, 4)\'\n\n    By default, larger matrices get collapsed into float arrays:\n    >>> print(glsl_code( Matrix([[1,2,3,4,5],[6,7,8,9,10]]) ))\n    float[10](\n       1, 2, 3, 4,  5,\n       6, 7, 8, 9, 10\n    ) /* a 2x5 matrix */\n\n    The type of array constructor used to print GLSL arrays can be controlled\n    via the ``array_type`` parameter:\n    >>> glsl_code(Matrix([1,2,3,4,5]), array_type=\'int\')\n    \'int[5](1, 2, 3, 4, 5)\'\n\n    Passing a list of strings or ``symbols`` to the ``assign_to`` parameter will yield\n    a multi-line assignment for each item in an array-like expression:\n    >>> x_struct_members = symbols(\'x.a x.b x.c x.d\')\n    >>> print(glsl_code(Matrix([1,2,3,4]), assign_to=x_struct_members))\n    x.a = 1;\n    x.b = 2;\n    x.c = 3;\n    x.d = 4;\n\n    This could be useful in cases where it\'s desirable to modify members of a\n    GLSL ``Struct``.  It could also be used to spread items from an array-like\n    expression into various miscellaneous assignments:\n    >>> misc_assignments = (\'x[0]\', \'x[1]\', \'float y\', \'float z\')\n    >>> print(glsl_code(Matrix([1,2,3,4]), assign_to=misc_assignments))\n    x[0] = 1;\n    x[1] = 2;\n    float y = 3;\n    float z = 4;\n\n    Passing ``mat_nested = True`` instead prints out nested float arrays, which are\n    supported in GLSL 4.3 and above.\n    >>> mat = Matrix([\n    ... [ 0,  1,  2],\n    ... [ 3,  4,  5],\n    ... [ 6,  7,  8],\n    ... [ 9, 10, 11],\n    ... [12, 13, 14]])\n    >>> print(glsl_code( mat, mat_nested = True ))\n    float[5][3](\n       float[]( 0,  1,  2),\n       float[]( 3,  4,  5),\n       float[]( 6,  7,  8),\n       float[]( 9, 10, 11),\n       float[](12, 13, 14)\n    )\n\n\n\n    Custom printing can be defined for certain types by passing a dictionary of\n    "type" : "function" to the ``user_functions`` kwarg. Alternatively, the\n    dictionary value can be a list of tuples i.e. [(argument_test,\n    js_function_string)].\n\n    >>> custom_functions = {\n    ...   "ceiling": "CEIL",\n    ...   "Abs": [(lambda x: not x.is_integer, "fabs"),\n    ...           (lambda x: x.is_integer, "ABS")]\n    ... }\n    >>> glsl_code(Abs(x) + ceiling(x), user_functions=custom_functions)\n    \'fabs(x) + CEIL(x)\'\n\n    If further control is needed, addition, subtraction, multiplication and\n    division operators can be replaced with ``add``, ``sub``, and ``mul``\n    functions.  This is done by passing ``use_operators = False``:\n\n    >>> x,y,z = symbols(\'x,y,z\')\n    >>> glsl_code(x*(y+z), use_operators = False)\n    \'mul(x, add(y, z))\'\n    >>> glsl_code(x*(y+z*(x-y)**z), use_operators = False)\n    \'mul(x, add(y, mul(z, pow(sub(x, y), z))))\'\n\n    ``Piecewise`` expressions are converted into conditionals. If an\n    ``assign_to`` variable is provided an if statement is created, otherwise\n    the ternary operator is used. Note that if the ``Piecewise`` lacks a\n    default term, represented by ``(expr, True)`` then an error will be thrown.\n    This is to prevent generating an expression that may not evaluate to\n    anything.\n\n    >>> from sympy import Piecewise\n    >>> expr = Piecewise((x + 1, x > 0), (x, True))\n    >>> print(glsl_code(expr, tau))\n    if (x > 0) {\n       tau = x + 1;\n    }\n    else {\n       tau = x;\n    }\n\n    Support for loops is provided through ``Indexed`` types. With\n    ``contract=True`` these expressions will be turned into loops, whereas\n    ``contract=False`` will just print the assignment expression that should be\n    looped over:\n\n    >>> from sympy import Eq, IndexedBase, Idx\n    >>> len_y = 5\n    >>> y = IndexedBase(\'y\', shape=(len_y,))\n    >>> t = IndexedBase(\'t\', shape=(len_y,))\n    >>> Dy = IndexedBase(\'Dy\', shape=(len_y-1,))\n    >>> i = Idx(\'i\', len_y-1)\n    >>> e=Eq(Dy[i], (y[i+1]-y[i])/(t[i+1]-t[i]))\n    >>> glsl_code(e.rhs, assign_to=e.lhs, contract=False)\n    \'Dy[i] = (y[i + 1] - y[i])/(t[i + 1] - t[i]);\'\n\n    >>> from sympy import Matrix, MatrixSymbol\n    >>> mat = Matrix([x**2, Piecewise((x + 1, x > 0), (x, True)), sin(x)])\n    >>> A = MatrixSymbol(\'A\', 3, 1)\n    >>> print(glsl_code(mat, A))\n    A[0][0] = pow(x, 2.0);\n    if (x > 0) {\n       A[1][0] = x + 1;\n    }\n    else {\n       A[1][0] = x;\n    }\n    A[2][0] = sin(x);\n    '
    return GLSLPrinter(settings).doprint(expr, assign_to)

def print_glsl(expr, **settings):
    if False:
        i = 10
        return i + 15
    'Prints the GLSL representation of the given expression.\n\n       See GLSLPrinter init function for settings.\n    '
    print(glsl_code(expr, **settings))