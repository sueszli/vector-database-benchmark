from sympy.printing import pycode, ccode, fcode
from sympy.external import import_module
from sympy.utilities.decorator import doctest_depends_on
lfortran = import_module('lfortran')
cin = import_module('clang.cindex', import_kwargs={'fromlist': ['cindex']})
if lfortran:
    from sympy.parsing.fortran.fortran_parser import src_to_sympy
if cin:
    from sympy.parsing.c.c_parser import parse_c

@doctest_depends_on(modules=['lfortran', 'clang.cindex'])
class SymPyExpression:
    """Class to store and handle SymPy expressions

    This class will hold SymPy Expressions and handle the API for the
    conversion to and from different languages.

    It works with the C and the Fortran Parser to generate SymPy expressions
    which are stored here and which can be converted to multiple language's
    source code.

    Notes
    =====

    The module and its API are currently under development and experimental
    and can be changed during development.

    The Fortran parser does not support numeric assignments, so all the
    variables have been Initialized to zero.

    The module also depends on external dependencies:

    - LFortran which is required to use the Fortran parser
    - Clang which is required for the C parser

    Examples
    ========

    Example of parsing C code:

    >>> from sympy.parsing.sym_expr import SymPyExpression
    >>> src = '''
    ... int a,b;
    ... float c = 2, d =4;
    ... '''
    >>> a = SymPyExpression(src, 'c')
    >>> a.return_expr()
    [Declaration(Variable(a, type=intc)),
    Declaration(Variable(b, type=intc)),
    Declaration(Variable(c, type=float32, value=2.0)),
    Declaration(Variable(d, type=float32, value=4.0))]

    An example of variable definition:

    >>> from sympy.parsing.sym_expr import SymPyExpression
    >>> src2 = '''
    ... integer :: a, b, c, d
    ... real :: p, q, r, s
    ... '''
    >>> p = SymPyExpression()
    >>> p.convert_to_expr(src2, 'f')
    >>> p.convert_to_c()
    ['int a = 0', 'int b = 0', 'int c = 0', 'int d = 0', 'double p = 0.0', 'double q = 0.0', 'double r = 0.0', 'double s = 0.0']

    An example of Assignment:

    >>> from sympy.parsing.sym_expr import SymPyExpression
    >>> src3 = '''
    ... integer :: a, b, c, d, e
    ... d = a + b - c
    ... e = b * d + c * e / a
    ... '''
    >>> p = SymPyExpression(src3, 'f')
    >>> p.convert_to_python()
    ['a = 0', 'b = 0', 'c = 0', 'd = 0', 'e = 0', 'd = a + b - c', 'e = b*d + c*e/a']

    An example of function definition:

    >>> from sympy.parsing.sym_expr import SymPyExpression
    >>> src = '''
    ... integer function f(a,b)
    ... integer, intent(in) :: a, b
    ... integer :: r
    ... end function
    ... '''
    >>> a = SymPyExpression(src, 'f')
    >>> a.convert_to_python()
    ['def f(a, b):\\n   f = 0\\n    r = 0\\n    return f']

    """

    def __init__(self, source_code=None, mode=None):
        if False:
            for i in range(10):
                print('nop')
        'Constructor for SymPyExpression class'
        super().__init__()
        if not (mode or source_code):
            self._expr = []
        elif mode:
            if source_code:
                if mode.lower() == 'f':
                    if lfortran:
                        self._expr = src_to_sympy(source_code)
                    else:
                        raise ImportError('LFortran is not installed, cannot parse Fortran code')
                elif mode.lower() == 'c':
                    if cin:
                        self._expr = parse_c(source_code)
                    else:
                        raise ImportError('Clang is not installed, cannot parse C code')
                else:
                    raise NotImplementedError('Parser for specified language is not implemented')
            else:
                raise ValueError('Source code not present')
        else:
            raise ValueError('Please specify a mode for conversion')

    def convert_to_expr(self, src_code, mode):
        if False:
            for i in range(10):
                print('nop')
        "Converts the given source code to SymPy Expressions\n\n        Attributes\n        ==========\n\n        src_code : String\n            the source code or filename of the source code that is to be\n            converted\n\n        mode: String\n            the mode to determine which parser is to be used according to\n            the language of the source code\n            f or F for Fortran\n            c or C for C/C++\n\n        Examples\n        ========\n\n        >>> from sympy.parsing.sym_expr import SymPyExpression\n        >>> src3 = '''\n        ... integer function f(a,b) result(r)\n        ... integer, intent(in) :: a, b\n        ... integer :: x\n        ... r = a + b -x\n        ... end function\n        ... '''\n        >>> p = SymPyExpression()\n        >>> p.convert_to_expr(src3, 'f')\n        >>> p.return_expr()\n        [FunctionDefinition(integer, name=f, parameters=(Variable(a), Variable(b)), body=CodeBlock(\n        Declaration(Variable(r, type=integer, value=0)),\n        Declaration(Variable(x, type=integer, value=0)),\n        Assignment(Variable(r), a + b - x),\n        Return(Variable(r))\n        ))]\n\n\n\n\n        "
        if mode.lower() == 'f':
            if lfortran:
                self._expr = src_to_sympy(src_code)
            else:
                raise ImportError('LFortran is not installed, cannot parse Fortran code')
        elif mode.lower() == 'c':
            if cin:
                self._expr = parse_c(src_code)
            else:
                raise ImportError('Clang is not installed, cannot parse C code')
        else:
            raise NotImplementedError('Parser for specified language has not been implemented')

    def convert_to_python(self):
        if False:
            for i in range(10):
                print('nop')
        "Returns a list with Python code for the SymPy expressions\n\n        Examples\n        ========\n\n        >>> from sympy.parsing.sym_expr import SymPyExpression\n        >>> src2 = '''\n        ... integer :: a, b, c, d\n        ... real :: p, q, r, s\n        ... c = a/b\n        ... d = c/a\n        ... s = p/q\n        ... r = q/p\n        ... '''\n        >>> p = SymPyExpression(src2, 'f')\n        >>> p.convert_to_python()\n        ['a = 0', 'b = 0', 'c = 0', 'd = 0', 'p = 0.0', 'q = 0.0', 'r = 0.0', 's = 0.0', 'c = a/b', 'd = c/a', 's = p/q', 'r = q/p']\n\n        "
        self._pycode = []
        for iter in self._expr:
            self._pycode.append(pycode(iter))
        return self._pycode

    def convert_to_c(self):
        if False:
            while True:
                i = 10
        "Returns a list with the c source code for the SymPy expressions\n\n\n        Examples\n        ========\n\n        >>> from sympy.parsing.sym_expr import SymPyExpression\n        >>> src2 = '''\n        ... integer :: a, b, c, d\n        ... real :: p, q, r, s\n        ... c = a/b\n        ... d = c/a\n        ... s = p/q\n        ... r = q/p\n        ... '''\n        >>> p = SymPyExpression()\n        >>> p.convert_to_expr(src2, 'f')\n        >>> p.convert_to_c()\n        ['int a = 0', 'int b = 0', 'int c = 0', 'int d = 0', 'double p = 0.0', 'double q = 0.0', 'double r = 0.0', 'double s = 0.0', 'c = a/b;', 'd = c/a;', 's = p/q;', 'r = q/p;']\n\n        "
        self._ccode = []
        for iter in self._expr:
            self._ccode.append(ccode(iter))
        return self._ccode

    def convert_to_fortran(self):
        if False:
            while True:
                i = 10
        "Returns a list with the fortran source code for the SymPy expressions\n\n        Examples\n        ========\n\n        >>> from sympy.parsing.sym_expr import SymPyExpression\n        >>> src2 = '''\n        ... integer :: a, b, c, d\n        ... real :: p, q, r, s\n        ... c = a/b\n        ... d = c/a\n        ... s = p/q\n        ... r = q/p\n        ... '''\n        >>> p = SymPyExpression(src2, 'f')\n        >>> p.convert_to_fortran()\n        ['      integer*4 a', '      integer*4 b', '      integer*4 c', '      integer*4 d', '      real*8 p', '      real*8 q', '      real*8 r', '      real*8 s', '      c = a/b', '      d = c/a', '      s = p/q', '      r = q/p']\n\n        "
        self._fcode = []
        for iter in self._expr:
            self._fcode.append(fcode(iter))
        return self._fcode

    def return_expr(self):
        if False:
            return 10
        "Returns the expression list\n\n        Examples\n        ========\n\n        >>> from sympy.parsing.sym_expr import SymPyExpression\n        >>> src3 = '''\n        ... integer function f(a,b)\n        ... integer, intent(in) :: a, b\n        ... integer :: r\n        ... r = a+b\n        ... f = r\n        ... end function\n        ... '''\n        >>> p = SymPyExpression()\n        >>> p.convert_to_expr(src3, 'f')\n        >>> p.return_expr()\n        [FunctionDefinition(integer, name=f, parameters=(Variable(a), Variable(b)), body=CodeBlock(\n        Declaration(Variable(f, type=integer, value=0)),\n        Declaration(Variable(r, type=integer, value=0)),\n        Assignment(Variable(f), Variable(r)),\n        Return(Variable(f))\n        ))]\n\n        "
        return self._expr