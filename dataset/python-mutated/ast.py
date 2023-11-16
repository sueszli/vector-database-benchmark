"""
Types used to represent a full function/module as an Abstract Syntax Tree.

Most types are small, and are merely used as tokens in the AST. A tree diagram
has been included below to illustrate the relationships between the AST types.


AST Type Tree
-------------
::

  *Basic*
       |
       |
   CodegenAST
       |
       |--->AssignmentBase
       |             |--->Assignment
       |             |--->AugmentedAssignment
       |                                    |--->AddAugmentedAssignment
       |                                    |--->SubAugmentedAssignment
       |                                    |--->MulAugmentedAssignment
       |                                    |--->DivAugmentedAssignment
       |                                    |--->ModAugmentedAssignment
       |
       |--->CodeBlock
       |
       |
       |--->Token
                |--->Attribute
                |--->For
                |--->String
                |       |--->QuotedString
                |       |--->Comment
                |--->Type
                |       |--->IntBaseType
                |       |              |--->_SizedIntType
                |       |                               |--->SignedIntType
                |       |                               |--->UnsignedIntType
                |       |--->FloatBaseType
                |                        |--->FloatType
                |                        |--->ComplexBaseType
                |                                           |--->ComplexType
                |--->Node
                |       |--->Variable
                |       |           |---> Pointer
                |       |--->FunctionPrototype
                |                            |--->FunctionDefinition
                |--->Element
                |--->Declaration
                |--->While
                |--->Scope
                |--->Stream
                |--->Print
                |--->FunctionCall
                |--->BreakToken
                |--->ContinueToken
                |--->NoneToken
                |--->Return


Predefined types
----------------

A number of ``Type`` instances are provided in the ``sympy.codegen.ast`` module
for convenience. Perhaps the two most common ones for code-generation (of numeric
codes) are ``float32`` and ``float64`` (known as single and double precision respectively).
There are also precision generic versions of Types (for which the codeprinters selects the
underlying data type at time of printing): ``real``, ``integer``, ``complex_``, ``bool_``.

The other ``Type`` instances defined are:

- ``intc``: Integer type used by C's "int".
- ``intp``: Integer type used by C's "unsigned".
- ``int8``, ``int16``, ``int32``, ``int64``: n-bit integers.
- ``uint8``, ``uint16``, ``uint32``, ``uint64``: n-bit unsigned integers.
- ``float80``: known as "extended precision" on modern x86/amd64 hardware.
- ``complex64``: Complex number represented by two ``float32`` numbers
- ``complex128``: Complex number represented by two ``float64`` numbers

Using the nodes
---------------

It is possible to construct simple algorithms using the AST nodes. Let's construct a loop applying
Newton's method::

    >>> from sympy import symbols, cos
    >>> from sympy.codegen.ast import While, Assignment, aug_assign, Print, QuotedString
    >>> t, dx, x = symbols('tol delta val')
    >>> expr = cos(x) - x**3
    >>> whl = While(abs(dx) > t, [
    ...     Assignment(dx, -expr/expr.diff(x)),
    ...     aug_assign(x, '+', dx),
    ...     Print([x])
    ... ])
    >>> from sympy import pycode
    >>> py_str = pycode(whl)
    >>> print(py_str)
    while (abs(delta) > tol):
        delta = (val**3 - math.cos(val))/(-3*val**2 - math.sin(val))
        val += delta
        print(val)
    >>> import math
    >>> tol, val, delta = 1e-5, 0.5, float('inf')
    >>> exec(py_str)
    1.1121416371
    0.909672693737
    0.867263818209
    0.865477135298
    0.865474033111
    >>> print('%3.1g' % (math.cos(val) - val**3))
    -3e-11

If we want to generate Fortran code for the same while loop we simple call ``fcode``::

    >>> from sympy import fcode
    >>> print(fcode(whl, standard=2003, source_format='free'))
    do while (abs(delta) > tol)
       delta = (val**3 - cos(val))/(-3*val**2 - sin(val))
       val = val + delta
       print *, val
    end do

There is a function constructing a loop (or a complete function) like this in
:mod:`sympy.codegen.algorithms`.

"""
from __future__ import annotations
from typing import Any
from collections import defaultdict
from sympy.core.relational import Ge, Gt, Le, Lt
from sympy.core import Symbol, Tuple, Dummy
from sympy.core.basic import Basic
from sympy.core.expr import Expr, Atom
from sympy.core.numbers import Float, Integer, oo
from sympy.core.sympify import _sympify, sympify, SympifyError
from sympy.utilities.iterables import iterable, topological_sort, numbered_symbols, filter_symbols

def _mk_Tuple(args):
    if False:
        while True:
            i = 10
    '\n    Create a SymPy Tuple object from an iterable, converting Python strings to\n    AST strings.\n\n    Parameters\n    ==========\n\n    args: iterable\n        Arguments to :class:`sympy.Tuple`.\n\n    Returns\n    =======\n\n    sympy.Tuple\n    '
    args = [String(arg) if isinstance(arg, str) else arg for arg in args]
    return Tuple(*args)

class CodegenAST(Basic):
    __slots__ = ()

class Token(CodegenAST):
    """ Base class for the AST types.

    Explanation
    ===========

    Defining fields are set in ``_fields``. Attributes (defined in _fields)
    are only allowed to contain instances of Basic (unless atomic, see
    ``String``). The arguments to ``__new__()`` correspond to the attributes in
    the order defined in ``_fields`. The ``defaults`` class attribute is a
    dictionary mapping attribute names to their default values.

    Subclasses should not need to override the ``__new__()`` method. They may
    define a class or static method named ``_construct_<attr>`` for each
    attribute to process the value passed to ``__new__()``. Attributes listed
    in the class attribute ``not_in_args`` are not passed to :class:`~.Basic`.
    """
    __slots__: tuple[str, ...] = ()
    _fields = __slots__
    defaults: dict[str, Any] = {}
    not_in_args: list[str] = []
    indented_args = ['body']

    @property
    def is_Atom(self):
        if False:
            print('Hello World!')
        return len(self._fields) == 0

    @classmethod
    def _get_constructor(cls, attr):
        if False:
            print('Hello World!')
        ' Get the constructor function for an attribute by name. '
        return getattr(cls, '_construct_%s' % attr, lambda x: x)

    @classmethod
    def _construct(cls, attr, arg):
        if False:
            print('Hello World!')
        ' Construct an attribute value from argument passed to ``__new__()``. '
        if arg == None:
            return cls.defaults.get(attr, none)
        elif isinstance(arg, Dummy):
            return arg
        else:
            return cls._get_constructor(attr)(arg)

    def __new__(cls, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        if len(args) == 1 and (not kwargs) and isinstance(args[0], cls):
            return args[0]
        if len(args) > len(cls._fields):
            raise ValueError('Too many arguments (%d), expected at most %d' % (len(args), len(cls._fields)))
        attrvals = []
        for (attrname, argval) in zip(cls._fields, args):
            if attrname in kwargs:
                raise TypeError('Got multiple values for attribute %r' % attrname)
            attrvals.append(cls._construct(attrname, argval))
        for attrname in cls._fields[len(args):]:
            if attrname in kwargs:
                argval = kwargs.pop(attrname)
            elif attrname in cls.defaults:
                argval = cls.defaults[attrname]
            else:
                raise TypeError('No value for %r given and attribute has no default' % attrname)
            attrvals.append(cls._construct(attrname, argval))
        if kwargs:
            raise ValueError('Unknown keyword arguments: %s' % ' '.join(kwargs))
        basic_args = [val for (attr, val) in zip(cls._fields, attrvals) if attr not in cls.not_in_args]
        obj = CodegenAST.__new__(cls, *basic_args)
        for (attr, arg) in zip(cls._fields, attrvals):
            setattr(obj, attr, arg)
        return obj

    def __eq__(self, other):
        if False:
            return 10
        if not isinstance(other, self.__class__):
            return False
        for attr in self._fields:
            if getattr(self, attr) != getattr(other, attr):
                return False
        return True

    def _hashable_content(self):
        if False:
            print('Hello World!')
        return tuple([getattr(self, attr) for attr in self._fields])

    def __hash__(self):
        if False:
            while True:
                i = 10
        return super().__hash__()

    def _joiner(self, k, indent_level):
        if False:
            for i in range(10):
                print('nop')
        return ',\n' + ' ' * indent_level if k in self.indented_args else ', '

    def _indented(self, printer, k, v, *args, **kwargs):
        if False:
            print('Hello World!')
        il = printer._context['indent_level']

        def _print(arg):
            if False:
                i = 10
                return i + 15
            if isinstance(arg, Token):
                return printer._print(arg, *args, joiner=self._joiner(k, il), **kwargs)
            else:
                return printer._print(arg, *args, **kwargs)
        if isinstance(v, Tuple):
            joined = self._joiner(k, il).join([_print(arg) for arg in v.args])
            if k in self.indented_args:
                return '(\n' + ' ' * il + joined + ',\n' + ' ' * (il - 4) + ')'
            else:
                return ('({0},)' if len(v.args) == 1 else '({0})').format(joined)
        else:
            return _print(v)

    def _sympyrepr(self, printer, *args, joiner=', ', **kwargs):
        if False:
            print('Hello World!')
        from sympy.printing.printer import printer_context
        exclude = kwargs.get('exclude', ())
        values = [getattr(self, k) for k in self._fields]
        indent_level = printer._context.get('indent_level', 0)
        arg_reprs = []
        for (i, (attr, value)) in enumerate(zip(self._fields, values)):
            if attr in exclude:
                continue
            if attr in self.defaults and value == self.defaults[attr]:
                continue
            ilvl = indent_level + 4 if attr in self.indented_args else 0
            with printer_context(printer, indent_level=ilvl):
                indented = self._indented(printer, attr, value, *args, **kwargs)
            arg_reprs.append(('{1}' if i == 0 else '{0}={1}').format(attr, indented.lstrip()))
        return '{}({})'.format(self.__class__.__name__, joiner.join(arg_reprs))
    _sympystr = _sympyrepr

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        from sympy.printing import srepr
        return srepr(self)

    def kwargs(self, exclude=(), apply=None):
        if False:
            print('Hello World!')
        " Get instance's attributes as dict of keyword arguments.\n\n        Parameters\n        ==========\n\n        exclude : collection of str\n            Collection of keywords to exclude.\n\n        apply : callable, optional\n            Function to apply to all values.\n        "
        kwargs = {k: getattr(self, k) for k in self._fields if k not in exclude}
        if apply is not None:
            return {k: apply(v) for (k, v) in kwargs.items()}
        else:
            return kwargs

class BreakToken(Token):
    """ Represents 'break' in C/Python ('exit' in Fortran).

    Use the premade instance ``break_`` or instantiate manually.

    Examples
    ========

    >>> from sympy import ccode, fcode
    >>> from sympy.codegen.ast import break_
    >>> ccode(break_)
    'break'
    >>> fcode(break_, source_format='free')
    'exit'
    """
break_ = BreakToken()

class ContinueToken(Token):
    """ Represents 'continue' in C/Python ('cycle' in Fortran)

    Use the premade instance ``continue_`` or instantiate manually.

    Examples
    ========

    >>> from sympy import ccode, fcode
    >>> from sympy.codegen.ast import continue_
    >>> ccode(continue_)
    'continue'
    >>> fcode(continue_, source_format='free')
    'cycle'
    """
continue_ = ContinueToken()

class NoneToken(Token):
    """ The AST equivalence of Python's NoneType

    The corresponding instance of Python's ``None`` is ``none``.

    Examples
    ========

    >>> from sympy.codegen.ast import none, Variable
    >>> from sympy import pycode
    >>> print(pycode(Variable('x').as_Declaration(value=none)))
    x = None

    """

    def __eq__(self, other):
        if False:
            while True:
                i = 10
        return other is None or isinstance(other, NoneToken)

    def _hashable_content(self):
        if False:
            print('Hello World!')
        return ()

    def __hash__(self):
        if False:
            while True:
                i = 10
        return super().__hash__()
none = NoneToken()

class AssignmentBase(CodegenAST):
    """ Abstract base class for Assignment and AugmentedAssignment.

    Attributes:
    ===========

    op : str
        Symbol for assignment operator, e.g. "=", "+=", etc.
    """

    def __new__(cls, lhs, rhs):
        if False:
            print('Hello World!')
        lhs = _sympify(lhs)
        rhs = _sympify(rhs)
        cls._check_args(lhs, rhs)
        return super().__new__(cls, lhs, rhs)

    @property
    def lhs(self):
        if False:
            return 10
        return self.args[0]

    @property
    def rhs(self):
        if False:
            print('Hello World!')
        return self.args[1]

    @classmethod
    def _check_args(cls, lhs, rhs):
        if False:
            i = 10
            return i + 15
        ' Check arguments to __new__ and raise exception if any problems found.\n\n        Derived classes may wish to override this.\n        '
        from sympy.matrices.expressions.matexpr import MatrixElement, MatrixSymbol
        from sympy.tensor.indexed import Indexed
        from sympy.tensor.array.expressions import ArrayElement
        assignable = (Symbol, MatrixSymbol, MatrixElement, Indexed, Element, Variable, ArrayElement)
        if not isinstance(lhs, assignable):
            raise TypeError('Cannot assign to lhs of type %s.' % type(lhs))
        lhs_is_mat = hasattr(lhs, 'shape') and (not isinstance(lhs, Indexed))
        rhs_is_mat = hasattr(rhs, 'shape') and (not isinstance(rhs, Indexed))
        if lhs_is_mat:
            if not rhs_is_mat:
                raise ValueError('Cannot assign a scalar to a matrix.')
            elif lhs.shape != rhs.shape:
                raise ValueError('Dimensions of lhs and rhs do not align.')
        elif rhs_is_mat and (not lhs_is_mat):
            raise ValueError('Cannot assign a matrix to a scalar.')

class Assignment(AssignmentBase):
    """
    Represents variable assignment for code generation.

    Parameters
    ==========

    lhs : Expr
        SymPy object representing the lhs of the expression. These should be
        singular objects, such as one would use in writing code. Notable types
        include Symbol, MatrixSymbol, MatrixElement, and Indexed. Types that
        subclass these types are also supported.

    rhs : Expr
        SymPy object representing the rhs of the expression. This can be any
        type, provided its shape corresponds to that of the lhs. For example,
        a Matrix type can be assigned to MatrixSymbol, but not to Symbol, as
        the dimensions will not align.

    Examples
    ========

    >>> from sympy import symbols, MatrixSymbol, Matrix
    >>> from sympy.codegen.ast import Assignment
    >>> x, y, z = symbols('x, y, z')
    >>> Assignment(x, y)
    Assignment(x, y)
    >>> Assignment(x, 0)
    Assignment(x, 0)
    >>> A = MatrixSymbol('A', 1, 3)
    >>> mat = Matrix([x, y, z]).T
    >>> Assignment(A, mat)
    Assignment(A, Matrix([[x, y, z]]))
    >>> Assignment(A[0, 1], x)
    Assignment(A[0, 1], x)
    """
    op = ':='

class AugmentedAssignment(AssignmentBase):
    """
    Base class for augmented assignments.

    Attributes:
    ===========

    binop : str
       Symbol for binary operation being applied in the assignment, such as "+",
       "*", etc.
    """
    binop = None

    @property
    def op(self):
        if False:
            print('Hello World!')
        return self.binop + '='

class AddAugmentedAssignment(AugmentedAssignment):
    binop = '+'

class SubAugmentedAssignment(AugmentedAssignment):
    binop = '-'

class MulAugmentedAssignment(AugmentedAssignment):
    binop = '*'

class DivAugmentedAssignment(AugmentedAssignment):
    binop = '/'

class ModAugmentedAssignment(AugmentedAssignment):
    binop = '%'
augassign_classes = {cls.binop: cls for cls in [AddAugmentedAssignment, SubAugmentedAssignment, MulAugmentedAssignment, DivAugmentedAssignment, ModAugmentedAssignment]}

def aug_assign(lhs, op, rhs):
    if False:
        return 10
    "\n    Create 'lhs op= rhs'.\n\n    Explanation\n    ===========\n\n    Represents augmented variable assignment for code generation. This is a\n    convenience function. You can also use the AugmentedAssignment classes\n    directly, like AddAugmentedAssignment(x, y).\n\n    Parameters\n    ==========\n\n    lhs : Expr\n        SymPy object representing the lhs of the expression. These should be\n        singular objects, such as one would use in writing code. Notable types\n        include Symbol, MatrixSymbol, MatrixElement, and Indexed. Types that\n        subclass these types are also supported.\n\n    op : str\n        Operator (+, -, /, \\*, %).\n\n    rhs : Expr\n        SymPy object representing the rhs of the expression. This can be any\n        type, provided its shape corresponds to that of the lhs. For example,\n        a Matrix type can be assigned to MatrixSymbol, but not to Symbol, as\n        the dimensions will not align.\n\n    Examples\n    ========\n\n    >>> from sympy import symbols\n    >>> from sympy.codegen.ast import aug_assign\n    >>> x, y = symbols('x, y')\n    >>> aug_assign(x, '+', y)\n    AddAugmentedAssignment(x, y)\n    "
    if op not in augassign_classes:
        raise ValueError('Unrecognized operator %s' % op)
    return augassign_classes[op](lhs, rhs)

class CodeBlock(CodegenAST):
    """
    Represents a block of code.

    Explanation
    ===========

    For now only assignments are supported. This restriction will be lifted in
    the future.

    Useful attributes on this object are:

    ``left_hand_sides``:
        Tuple of left-hand sides of assignments, in order.
    ``left_hand_sides``:
        Tuple of right-hand sides of assignments, in order.
    ``free_symbols``: Free symbols of the expressions in the right-hand sides
        which do not appear in the left-hand side of an assignment.

    Useful methods on this object are:

    ``topological_sort``:
        Class method. Return a CodeBlock with assignments
        sorted so that variables are assigned before they
        are used.
    ``cse``:
        Return a new CodeBlock with common subexpressions eliminated and
        pulled out as assignments.

    Examples
    ========

    >>> from sympy import symbols, ccode
    >>> from sympy.codegen.ast import CodeBlock, Assignment
    >>> x, y = symbols('x y')
    >>> c = CodeBlock(Assignment(x, 1), Assignment(y, x + 1))
    >>> print(ccode(c))
    x = 1;
    y = x + 1;

    """

    def __new__(cls, *args):
        if False:
            i = 10
            return i + 15
        left_hand_sides = []
        right_hand_sides = []
        for i in args:
            if isinstance(i, Assignment):
                (lhs, rhs) = i.args
                left_hand_sides.append(lhs)
                right_hand_sides.append(rhs)
        obj = CodegenAST.__new__(cls, *args)
        obj.left_hand_sides = Tuple(*left_hand_sides)
        obj.right_hand_sides = Tuple(*right_hand_sides)
        return obj

    def __iter__(self):
        if False:
            print('Hello World!')
        return iter(self.args)

    def _sympyrepr(self, printer, *args, **kwargs):
        if False:
            return 10
        il = printer._context.get('indent_level', 0)
        joiner = ',\n' + ' ' * il
        joined = joiner.join(map(printer._print, self.args))
        return '{}(\n'.format(' ' * (il - 4) + self.__class__.__name__) + ' ' * il + joined + '\n' + ' ' * (il - 4) + ')'
    _sympystr = _sympyrepr

    @property
    def free_symbols(self):
        if False:
            i = 10
            return i + 15
        return super().free_symbols - set(self.left_hand_sides)

    @classmethod
    def topological_sort(cls, assignments):
        if False:
            while True:
                i = 10
        "\n        Return a CodeBlock with topologically sorted assignments so that\n        variables are assigned before they are used.\n\n        Examples\n        ========\n\n        The existing order of assignments is preserved as much as possible.\n\n        This function assumes that variables are assigned to only once.\n\n        This is a class constructor so that the default constructor for\n        CodeBlock can error when variables are used before they are assigned.\n\n        >>> from sympy import symbols\n        >>> from sympy.codegen.ast import CodeBlock, Assignment\n        >>> x, y, z = symbols('x y z')\n\n        >>> assignments = [\n        ...     Assignment(x, y + z),\n        ...     Assignment(y, z + 1),\n        ...     Assignment(z, 2),\n        ... ]\n        >>> CodeBlock.topological_sort(assignments)\n        CodeBlock(\n            Assignment(z, 2),\n            Assignment(y, z + 1),\n            Assignment(x, y + z)\n        )\n\n        "
        if not all((isinstance(i, Assignment) for i in assignments)):
            raise NotImplementedError('CodeBlock.topological_sort only supports Assignments')
        if any((isinstance(i, AugmentedAssignment) for i in assignments)):
            raise NotImplementedError('CodeBlock.topological_sort does not yet work with AugmentedAssignments')
        A = list(enumerate(assignments))
        var_map = defaultdict(list)
        for node in A:
            (i, a) = node
            var_map[a.lhs].append(node)
        E = []
        for dst_node in A:
            (i, a) = dst_node
            for s in a.rhs.free_symbols:
                for src_node in var_map[s]:
                    E.append((src_node, dst_node))
        ordered_assignments = topological_sort([A, E])
        return cls(*[a for (i, a) in ordered_assignments])

    def cse(self, symbols=None, optimizations=None, postprocess=None, order='canonical'):
        if False:
            i = 10
            return i + 15
        "\n        Return a new code block with common subexpressions eliminated.\n\n        Explanation\n        ===========\n\n        See the docstring of :func:`sympy.simplify.cse_main.cse` for more\n        information.\n\n        Examples\n        ========\n\n        >>> from sympy import symbols, sin\n        >>> from sympy.codegen.ast import CodeBlock, Assignment\n        >>> x, y, z = symbols('x y z')\n\n        >>> c = CodeBlock(\n        ...     Assignment(x, 1),\n        ...     Assignment(y, sin(x) + 1),\n        ...     Assignment(z, sin(x) - 1),\n        ... )\n        ...\n        >>> c.cse()\n        CodeBlock(\n            Assignment(x, 1),\n            Assignment(x0, sin(x)),\n            Assignment(y, x0 + 1),\n            Assignment(z, x0 - 1)\n        )\n\n        "
        from sympy.simplify.cse_main import cse
        if not all((isinstance(i, Assignment) for i in self.args)):
            raise NotImplementedError('CodeBlock.cse only supports Assignments')
        if any((isinstance(i, AugmentedAssignment) for i in self.args)):
            raise NotImplementedError('CodeBlock.cse does not yet work with AugmentedAssignments')
        for (i, lhs) in enumerate(self.left_hand_sides):
            if lhs in self.left_hand_sides[:i]:
                raise NotImplementedError('Duplicate assignments to the same variable are not yet supported (%s)' % lhs)
        existing_symbols = self.atoms(Symbol)
        if symbols is None:
            symbols = numbered_symbols()
        symbols = filter_symbols(symbols, existing_symbols)
        (replacements, reduced_exprs) = cse(list(self.right_hand_sides), symbols=symbols, optimizations=optimizations, postprocess=postprocess, order=order)
        new_block = [Assignment(var, expr) for (var, expr) in zip(self.left_hand_sides, reduced_exprs)]
        new_assignments = [Assignment(var, expr) for (var, expr) in replacements]
        return self.topological_sort(new_assignments + new_block)

class For(Token):
    """Represents a 'for-loop' in the code.

    Expressions are of the form:
        "for target in iter:
            body..."

    Parameters
    ==========

    target : symbol
    iter : iterable
    body : CodeBlock or iterable
!        When passed an iterable it is used to instantiate a CodeBlock.

    Examples
    ========

    >>> from sympy import symbols, Range
    >>> from sympy.codegen.ast import aug_assign, For
    >>> x, i, j, k = symbols('x i j k')
    >>> for_i = For(i, Range(10), [aug_assign(x, '+', i*j*k)])
    >>> for_i  # doctest: -NORMALIZE_WHITESPACE
    For(i, iterable=Range(0, 10, 1), body=CodeBlock(
        AddAugmentedAssignment(x, i*j*k)
    ))
    >>> for_ji = For(j, Range(7), [for_i])
    >>> for_ji  # doctest: -NORMALIZE_WHITESPACE
    For(j, iterable=Range(0, 7, 1), body=CodeBlock(
        For(i, iterable=Range(0, 10, 1), body=CodeBlock(
            AddAugmentedAssignment(x, i*j*k)
        ))
    ))
    >>> for_kji =For(k, Range(5), [for_ji])
    >>> for_kji  # doctest: -NORMALIZE_WHITESPACE
    For(k, iterable=Range(0, 5, 1), body=CodeBlock(
        For(j, iterable=Range(0, 7, 1), body=CodeBlock(
            For(i, iterable=Range(0, 10, 1), body=CodeBlock(
                AddAugmentedAssignment(x, i*j*k)
            ))
        ))
    ))
    """
    __slots__ = _fields = ('target', 'iterable', 'body')
    _construct_target = staticmethod(_sympify)

    @classmethod
    def _construct_body(cls, itr):
        if False:
            while True:
                i = 10
        if isinstance(itr, CodeBlock):
            return itr
        else:
            return CodeBlock(*itr)

    @classmethod
    def _construct_iterable(cls, itr):
        if False:
            i = 10
            return i + 15
        if not iterable(itr):
            raise TypeError('iterable must be an iterable')
        if isinstance(itr, list):
            itr = tuple(itr)
        return _sympify(itr)

class String(Atom, Token):
    """ SymPy object representing a string.

    Atomic object which is not an expression (as opposed to Symbol).

    Parameters
    ==========

    text : str

    Examples
    ========

    >>> from sympy.codegen.ast import String
    >>> f = String('foo')
    >>> f
    foo
    >>> str(f)
    'foo'
    >>> f.text
    'foo'
    >>> print(repr(f))
    String('foo')

    """
    __slots__ = _fields = ('text',)
    not_in_args = ['text']
    is_Atom = True

    @classmethod
    def _construct_text(cls, text):
        if False:
            return 10
        if not isinstance(text, str):
            raise TypeError('Argument text is not a string type.')
        return text

    def _sympystr(self, printer, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        return self.text

    def kwargs(self, exclude=(), apply=None):
        if False:
            while True:
                i = 10
        return {}

    @property
    def func(self):
        if False:
            while True:
                i = 10
        return lambda : self

    def _latex(self, printer):
        if False:
            for i in range(10):
                print('nop')
        from sympy.printing.latex import latex_escape
        return '\\texttt{{"{}"}}'.format(latex_escape(self.text))

class QuotedString(String):
    """ Represents a string which should be printed with quotes. """

class Comment(String):
    """ Represents a comment. """

class Node(Token):
    """ Subclass of Token, carrying the attribute 'attrs' (Tuple)

    Examples
    ========

    >>> from sympy.codegen.ast import Node, value_const, pointer_const
    >>> n1 = Node([value_const])
    >>> n1.attr_params('value_const')  # get the parameters of attribute (by name)
    ()
    >>> from sympy.codegen.fnodes import dimension
    >>> n2 = Node([value_const, dimension(5, 3)])
    >>> n2.attr_params(value_const)  # get the parameters of attribute (by Attribute instance)
    ()
    >>> n2.attr_params('dimension')  # get the parameters of attribute (by name)
    (5, 3)
    >>> n2.attr_params(pointer_const) is None
    True

    """
    __slots__: tuple[str, ...] = ('attrs',)
    _fields = __slots__
    defaults: dict[str, Any] = {'attrs': Tuple()}
    _construct_attrs = staticmethod(_mk_Tuple)

    def attr_params(self, looking_for):
        if False:
            print('Hello World!')
        ' Returns the parameters of the Attribute with name ``looking_for`` in self.attrs '
        for attr in self.attrs:
            if str(attr.name) == str(looking_for):
                return attr.parameters

class Type(Token):
    """ Represents a type.

    Explanation
    ===========

    The naming is a super-set of NumPy naming. Type has a classmethod
    ``from_expr`` which offer type deduction. It also has a method
    ``cast_check`` which casts the argument to its type, possibly raising an
    exception if rounding error is not within tolerances, or if the value is not
    representable by the underlying data type (e.g. unsigned integers).

    Parameters
    ==========

    name : str
        Name of the type, e.g. ``object``, ``int16``, ``float16`` (where the latter two
        would use the ``Type`` sub-classes ``IntType`` and ``FloatType`` respectively).
        If a ``Type`` instance is given, the said instance is returned.

    Examples
    ========

    >>> from sympy.codegen.ast import Type
    >>> t = Type.from_expr(42)
    >>> t
    integer
    >>> print(repr(t))
    IntBaseType(String('integer'))
    >>> from sympy.codegen.ast import uint8
    >>> uint8.cast_check(-1)   # doctest: +ELLIPSIS
    Traceback (most recent call last):
      ...
    ValueError: Minimum value for data type bigger than new value.
    >>> from sympy.codegen.ast import float32
    >>> v6 = 0.123456
    >>> float32.cast_check(v6)
    0.123456
    >>> v10 = 12345.67894
    >>> float32.cast_check(v10)  # doctest: +ELLIPSIS
    Traceback (most recent call last):
      ...
    ValueError: Casting gives a significantly different value.
    >>> boost_mp50 = Type('boost::multiprecision::cpp_dec_float_50')
    >>> from sympy import cxxcode
    >>> from sympy.codegen.ast import Declaration, Variable
    >>> cxxcode(Declaration(Variable('x', type=boost_mp50)))
    'boost::multiprecision::cpp_dec_float_50 x'

    References
    ==========

    .. [1] https://numpy.org/doc/stable/user/basics.types.html

    """
    __slots__: tuple[str, ...] = ('name',)
    _fields = __slots__
    _construct_name = String

    def _sympystr(self, printer, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        return str(self.name)

    @classmethod
    def from_expr(cls, expr):
        if False:
            print('Hello World!')
        " Deduces type from an expression or a ``Symbol``.\n\n        Parameters\n        ==========\n\n        expr : number or SymPy object\n            The type will be deduced from type or properties.\n\n        Examples\n        ========\n\n        >>> from sympy.codegen.ast import Type, integer, complex_\n        >>> Type.from_expr(2) == integer\n        True\n        >>> from sympy import Symbol\n        >>> Type.from_expr(Symbol('z', complex=True)) == complex_\n        True\n        >>> Type.from_expr(sum)  # doctest: +ELLIPSIS\n        Traceback (most recent call last):\n          ...\n        ValueError: Could not deduce type from expr.\n\n        Raises\n        ======\n\n        ValueError when type deduction fails.\n\n        "
        if isinstance(expr, (float, Float)):
            return real
        if isinstance(expr, (int, Integer)) or getattr(expr, 'is_integer', False):
            return integer
        if getattr(expr, 'is_real', False):
            return real
        if isinstance(expr, complex) or getattr(expr, 'is_complex', False):
            return complex_
        if isinstance(expr, bool) or getattr(expr, 'is_Relational', False):
            return bool_
        else:
            raise ValueError('Could not deduce type from expr.')

    def _check(self, value):
        if False:
            for i in range(10):
                print('nop')
        pass

    def cast_check(self, value, rtol=None, atol=0, precision_targets=None):
        if False:
            i = 10
            return i + 15
        " Casts a value to the data type of the instance.\n\n        Parameters\n        ==========\n\n        value : number\n        rtol : floating point number\n            Relative tolerance. (will be deduced if not given).\n        atol : floating point number\n            Absolute tolerance (in addition to ``rtol``).\n        type_aliases : dict\n            Maps substitutions for Type, e.g. {integer: int64, real: float32}\n\n        Examples\n        ========\n\n        >>> from sympy.codegen.ast import integer, float32, int8\n        >>> integer.cast_check(3.0) == 3\n        True\n        >>> float32.cast_check(1e-40)  # doctest: +ELLIPSIS\n        Traceback (most recent call last):\n          ...\n        ValueError: Minimum value for data type bigger than new value.\n        >>> int8.cast_check(256)  # doctest: +ELLIPSIS\n        Traceback (most recent call last):\n          ...\n        ValueError: Maximum value for data type smaller than new value.\n        >>> v10 = 12345.67894\n        >>> float32.cast_check(v10)  # doctest: +ELLIPSIS\n        Traceback (most recent call last):\n          ...\n        ValueError: Casting gives a significantly different value.\n        >>> from sympy.codegen.ast import float64\n        >>> float64.cast_check(v10)\n        12345.67894\n        >>> from sympy import Float\n        >>> v18 = Float('0.123456789012345646')\n        >>> float64.cast_check(v18)\n        Traceback (most recent call last):\n          ...\n        ValueError: Casting gives a significantly different value.\n        >>> from sympy.codegen.ast import float80\n        >>> float80.cast_check(v18)\n        0.123456789012345649\n\n        "
        val = sympify(value)
        ten = Integer(10)
        exp10 = getattr(self, 'decimal_dig', None)
        if rtol is None:
            rtol = 1e-15 if exp10 is None else 2.0 * ten ** (-exp10)

        def tol(num):
            if False:
                while True:
                    i = 10
            return atol + rtol * abs(num)
        new_val = self.cast_nocheck(value)
        self._check(new_val)
        delta = new_val - val
        if abs(delta) > tol(val):
            raise ValueError('Casting gives a significantly different value.')
        return new_val

    def _latex(self, printer):
        if False:
            return 10
        from sympy.printing.latex import latex_escape
        type_name = latex_escape(self.__class__.__name__)
        name = latex_escape(self.name.text)
        return '\\text{{{}}}\\left(\\texttt{{{}}}\\right)'.format(type_name, name)

class IntBaseType(Type):
    """ Integer base type, contains no size information. """
    __slots__ = ()
    cast_nocheck = lambda self, i: Integer(int(i))

class _SizedIntType(IntBaseType):
    __slots__ = ('nbits',)
    _fields = Type._fields + __slots__
    _construct_nbits = Integer

    def _check(self, value):
        if False:
            while True:
                i = 10
        if value < self.min:
            raise ValueError('Value is too small: %d < %d' % (value, self.min))
        if value > self.max:
            raise ValueError('Value is too big: %d > %d' % (value, self.max))

class SignedIntType(_SizedIntType):
    """ Represents a signed integer type. """
    __slots__ = ()

    @property
    def min(self):
        if False:
            i = 10
            return i + 15
        return -2 ** (self.nbits - 1)

    @property
    def max(self):
        if False:
            for i in range(10):
                print('nop')
        return 2 ** (self.nbits - 1) - 1

class UnsignedIntType(_SizedIntType):
    """ Represents an unsigned integer type. """
    __slots__ = ()

    @property
    def min(self):
        if False:
            i = 10
            return i + 15
        return 0

    @property
    def max(self):
        if False:
            return 10
        return 2 ** self.nbits - 1
two = Integer(2)

class FloatBaseType(Type):
    """ Represents a floating point number type. """
    __slots__ = ()
    cast_nocheck = Float

class FloatType(FloatBaseType):
    """ Represents a floating point type with fixed bit width.

    Base 2 & one sign bit is assumed.

    Parameters
    ==========

    name : str
        Name of the type.
    nbits : integer
        Number of bits used (storage).
    nmant : integer
        Number of bits used to represent the mantissa.
    nexp : integer
        Number of bits used to represent the mantissa.

    Examples
    ========

    >>> from sympy import S
    >>> from sympy.codegen.ast import FloatType
    >>> half_precision = FloatType('f16', nbits=16, nmant=10, nexp=5)
    >>> half_precision.max
    65504
    >>> half_precision.tiny == S(2)**-14
    True
    >>> half_precision.eps == S(2)**-10
    True
    >>> half_precision.dig == 3
    True
    >>> half_precision.decimal_dig == 5
    True
    >>> half_precision.cast_check(1.0)
    1.0
    >>> half_precision.cast_check(1e5)  # doctest: +ELLIPSIS
    Traceback (most recent call last):
      ...
    ValueError: Maximum value for data type smaller than new value.
    """
    __slots__ = ('nbits', 'nmant', 'nexp')
    _fields = Type._fields + __slots__
    _construct_nbits = _construct_nmant = _construct_nexp = Integer

    @property
    def max_exponent(self):
        if False:
            return 10
        ' The largest positive number n, such that 2**(n - 1) is a representable finite value. '
        return two ** (self.nexp - 1)

    @property
    def min_exponent(self):
        if False:
            print('Hello World!')
        ' The lowest negative number n, such that 2**(n - 1) is a valid normalized number. '
        return 3 - self.max_exponent

    @property
    def max(self):
        if False:
            return 10
        ' Maximum value representable. '
        return (1 - two ** (-(self.nmant + 1))) * two ** self.max_exponent

    @property
    def tiny(self):
        if False:
            for i in range(10):
                print('nop')
        ' The minimum positive normalized value. '
        return two ** (self.min_exponent - 1)

    @property
    def eps(self):
        if False:
            while True:
                i = 10
        ' Difference between 1.0 and the next representable value. '
        return two ** (-self.nmant)

    @property
    def dig(self):
        if False:
            print('Hello World!')
        ' Number of decimal digits that are guaranteed to be preserved in text.\n\n        When converting text -> float -> text, you are guaranteed that at least ``dig``\n        number of digits are preserved with respect to rounding or overflow.\n        '
        from sympy.functions import floor, log
        return floor(self.nmant * log(2) / log(10))

    @property
    def decimal_dig(self):
        if False:
            return 10
        ' Number of digits needed to store & load without loss.\n\n        Explanation\n        ===========\n\n        Number of decimal digits needed to guarantee that two consecutive conversions\n        (float -> text -> float) to be idempotent. This is useful when one do not want\n        to loose precision due to rounding errors when storing a floating point value\n        as text.\n        '
        from sympy.functions import ceiling, log
        return ceiling((self.nmant + 1) * log(2) / log(10) + 1)

    def cast_nocheck(self, value):
        if False:
            print('Hello World!')
        ' Casts without checking if out of bounds or subnormal. '
        if value == oo:
            return float(oo)
        elif value == -oo:
            return float(-oo)
        return Float(str(sympify(value).evalf(self.decimal_dig)), self.decimal_dig)

    def _check(self, value):
        if False:
            while True:
                i = 10
        if value < -self.max:
            raise ValueError('Value is too small: %d < %d' % (value, -self.max))
        if value > self.max:
            raise ValueError('Value is too big: %d > %d' % (value, self.max))
        if abs(value) < self.tiny:
            raise ValueError('Smallest (absolute) value for data type bigger than new value.')

class ComplexBaseType(FloatBaseType):
    __slots__ = ()

    def cast_nocheck(self, value):
        if False:
            print('Hello World!')
        ' Casts without checking if out of bounds or subnormal. '
        from sympy.functions import re, im
        return super().cast_nocheck(re(value)) + super().cast_nocheck(im(value)) * 1j

    def _check(self, value):
        if False:
            for i in range(10):
                print('nop')
        from sympy.functions import re, im
        super()._check(re(value))
        super()._check(im(value))

class ComplexType(ComplexBaseType, FloatType):
    """ Represents a complex floating point number. """
    __slots__ = ()
intc = IntBaseType('intc')
intp = IntBaseType('intp')
int8 = SignedIntType('int8', 8)
int16 = SignedIntType('int16', 16)
int32 = SignedIntType('int32', 32)
int64 = SignedIntType('int64', 64)
uint8 = UnsignedIntType('uint8', 8)
uint16 = UnsignedIntType('uint16', 16)
uint32 = UnsignedIntType('uint32', 32)
uint64 = UnsignedIntType('uint64', 64)
float16 = FloatType('float16', 16, nexp=5, nmant=10)
float32 = FloatType('float32', 32, nexp=8, nmant=23)
float64 = FloatType('float64', 64, nexp=11, nmant=52)
float80 = FloatType('float80', 80, nexp=15, nmant=63)
float128 = FloatType('float128', 128, nexp=15, nmant=112)
float256 = FloatType('float256', 256, nexp=19, nmant=236)
complex64 = ComplexType('complex64', nbits=64, **float32.kwargs(exclude=('name', 'nbits')))
complex128 = ComplexType('complex128', nbits=128, **float64.kwargs(exclude=('name', 'nbits')))
untyped = Type('untyped')
real = FloatBaseType('real')
integer = IntBaseType('integer')
complex_ = ComplexBaseType('complex')
bool_ = Type('bool')

class Attribute(Token):
    """ Attribute (possibly parametrized)

    For use with :class:`sympy.codegen.ast.Node` (which takes instances of
    ``Attribute`` as ``attrs``).

    Parameters
    ==========

    name : str
    parameters : Tuple

    Examples
    ========

    >>> from sympy.codegen.ast import Attribute
    >>> volatile = Attribute('volatile')
    >>> volatile
    volatile
    >>> print(repr(volatile))
    Attribute(String('volatile'))
    >>> a = Attribute('foo', [1, 2, 3])
    >>> a
    foo(1, 2, 3)
    >>> a.parameters == (1, 2, 3)
    True
    """
    __slots__ = _fields = ('name', 'parameters')
    defaults = {'parameters': Tuple()}
    _construct_name = String
    _construct_parameters = staticmethod(_mk_Tuple)

    def _sympystr(self, printer, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        result = str(self.name)
        if self.parameters:
            result += '(%s)' % ', '.join((printer._print(arg, *args, **kwargs) for arg in self.parameters))
        return result
value_const = Attribute('value_const')
pointer_const = Attribute('pointer_const')

class Variable(Node):
    """ Represents a variable.

    Parameters
    ==========

    symbol : Symbol
    type : Type (optional)
        Type of the variable.
    attrs : iterable of Attribute instances
        Will be stored as a Tuple.

    Examples
    ========

    >>> from sympy import Symbol
    >>> from sympy.codegen.ast import Variable, float32, integer
    >>> x = Symbol('x')
    >>> v = Variable(x, type=float32)
    >>> v.attrs
    ()
    >>> v == Variable('x')
    False
    >>> v == Variable('x', type=float32)
    True
    >>> v
    Variable(x, type=float32)

    One may also construct a ``Variable`` instance with the type deduced from
    assumptions about the symbol using the ``deduced`` classmethod:

    >>> i = Symbol('i', integer=True)
    >>> v = Variable.deduced(i)
    >>> v.type == integer
    True
    >>> v == Variable('i')
    False
    >>> from sympy.codegen.ast import value_const
    >>> value_const in v.attrs
    False
    >>> w = Variable('w', attrs=[value_const])
    >>> w
    Variable(w, attrs=(value_const,))
    >>> value_const in w.attrs
    True
    >>> w.as_Declaration(value=42)
    Declaration(Variable(w, value=42, attrs=(value_const,)))

    """
    __slots__ = ('symbol', 'type', 'value')
    _fields = __slots__ + Node._fields
    defaults = Node.defaults.copy()
    defaults.update({'type': untyped, 'value': none})
    _construct_symbol = staticmethod(sympify)
    _construct_value = staticmethod(sympify)

    @classmethod
    def deduced(cls, symbol, value=None, attrs=Tuple(), cast_check=True):
        if False:
            while True:
                i = 10
        " Alt. constructor with type deduction from ``Type.from_expr``.\n\n        Deduces type primarily from ``symbol``, secondarily from ``value``.\n\n        Parameters\n        ==========\n\n        symbol : Symbol\n        value : expr\n            (optional) value of the variable.\n        attrs : iterable of Attribute instances\n        cast_check : bool\n            Whether to apply ``Type.cast_check`` on ``value``.\n\n        Examples\n        ========\n\n        >>> from sympy import Symbol\n        >>> from sympy.codegen.ast import Variable, complex_\n        >>> n = Symbol('n', integer=True)\n        >>> str(Variable.deduced(n).type)\n        'integer'\n        >>> x = Symbol('x', real=True)\n        >>> v = Variable.deduced(x)\n        >>> v.type\n        real\n        >>> z = Symbol('z', complex=True)\n        >>> Variable.deduced(z).type == complex_\n        True\n\n        "
        if isinstance(symbol, Variable):
            return symbol
        try:
            type_ = Type.from_expr(symbol)
        except ValueError:
            type_ = Type.from_expr(value)
        if value is not None and cast_check:
            value = type_.cast_check(value)
        return cls(symbol, type=type_, value=value, attrs=attrs)

    def as_Declaration(self, **kwargs):
        if False:
            i = 10
            return i + 15
        " Convenience method for creating a Declaration instance.\n\n        Explanation\n        ===========\n\n        If the variable of the Declaration need to wrap a modified\n        variable keyword arguments may be passed (overriding e.g.\n        the ``value`` of the Variable instance).\n\n        Examples\n        ========\n\n        >>> from sympy.codegen.ast import Variable, NoneToken\n        >>> x = Variable('x')\n        >>> decl1 = x.as_Declaration()\n        >>> # value is special NoneToken() which must be tested with == operator\n        >>> decl1.variable.value is None  # won't work\n        False\n        >>> decl1.variable.value == None  # not PEP-8 compliant\n        True\n        >>> decl1.variable.value == NoneToken()  # OK\n        True\n        >>> decl2 = x.as_Declaration(value=42.0)\n        >>> decl2.variable.value == 42.0\n        True\n\n        "
        kw = self.kwargs()
        kw.update(kwargs)
        return Declaration(self.func(**kw))

    def _relation(self, rhs, op):
        if False:
            for i in range(10):
                print('nop')
        try:
            rhs = _sympify(rhs)
        except SympifyError:
            raise TypeError('Invalid comparison %s < %s' % (self, rhs))
        return op(self, rhs, evaluate=False)
    __lt__ = lambda self, other: self._relation(other, Lt)
    __le__ = lambda self, other: self._relation(other, Le)
    __ge__ = lambda self, other: self._relation(other, Ge)
    __gt__ = lambda self, other: self._relation(other, Gt)

class Pointer(Variable):
    """ Represents a pointer. See ``Variable``.

    Examples
    ========

    Can create instances of ``Element``:

    >>> from sympy import Symbol
    >>> from sympy.codegen.ast import Pointer
    >>> i = Symbol('i', integer=True)
    >>> p = Pointer('x')
    >>> p[i+1]
    Element(x, indices=(i + 1,))

    """
    __slots__ = ()

    def __getitem__(self, key):
        if False:
            print('Hello World!')
        try:
            return Element(self.symbol, key)
        except TypeError:
            return Element(self.symbol, (key,))

class Element(Token):
    """ Element in (a possibly N-dimensional) array.

    Examples
    ========

    >>> from sympy.codegen.ast import Element
    >>> elem = Element('x', 'ijk')
    >>> elem.symbol.name == 'x'
    True
    >>> elem.indices
    (i, j, k)
    >>> from sympy import ccode
    >>> ccode(elem)
    'x[i][j][k]'
    >>> ccode(Element('x', 'ijk', strides='lmn', offset='o'))
    'x[i*l + j*m + k*n + o]'

    """
    __slots__ = _fields = ('symbol', 'indices', 'strides', 'offset')
    defaults = {'strides': none, 'offset': none}
    _construct_symbol = staticmethod(sympify)
    _construct_indices = staticmethod(lambda arg: Tuple(*arg))
    _construct_strides = staticmethod(lambda arg: Tuple(*arg))
    _construct_offset = staticmethod(sympify)

class Declaration(Token):
    """ Represents a variable declaration

    Parameters
    ==========

    variable : Variable

    Examples
    ========

    >>> from sympy.codegen.ast import Declaration, NoneToken, untyped
    >>> z = Declaration('z')
    >>> z.variable.type == untyped
    True
    >>> # value is special NoneToken() which must be tested with == operator
    >>> z.variable.value is None  # won't work
    False
    >>> z.variable.value == None  # not PEP-8 compliant
    True
    >>> z.variable.value == NoneToken()  # OK
    True
    """
    __slots__ = _fields = ('variable',)
    _construct_variable = Variable

class While(Token):
    """ Represents a 'for-loop' in the code.

    Expressions are of the form:
        "while condition:
             body..."

    Parameters
    ==========

    condition : expression convertible to Boolean
    body : CodeBlock or iterable
        When passed an iterable it is used to instantiate a CodeBlock.

    Examples
    ========

    >>> from sympy import symbols, Gt, Abs
    >>> from sympy.codegen import aug_assign, Assignment, While
    >>> x, dx = symbols('x dx')
    >>> expr = 1 - x**2
    >>> whl = While(Gt(Abs(dx), 1e-9), [
    ...     Assignment(dx, -expr/expr.diff(x)),
    ...     aug_assign(x, '+', dx)
    ... ])

    """
    __slots__ = _fields = ('condition', 'body')
    _construct_condition = staticmethod(lambda cond: _sympify(cond))

    @classmethod
    def _construct_body(cls, itr):
        if False:
            while True:
                i = 10
        if isinstance(itr, CodeBlock):
            return itr
        else:
            return CodeBlock(*itr)

class Scope(Token):
    """ Represents a scope in the code.

    Parameters
    ==========

    body : CodeBlock or iterable
        When passed an iterable it is used to instantiate a CodeBlock.

    """
    __slots__ = _fields = ('body',)

    @classmethod
    def _construct_body(cls, itr):
        if False:
            print('Hello World!')
        if isinstance(itr, CodeBlock):
            return itr
        else:
            return CodeBlock(*itr)

class Stream(Token):
    """ Represents a stream.

    There are two predefined Stream instances ``stdout`` & ``stderr``.

    Parameters
    ==========

    name : str

    Examples
    ========

    >>> from sympy import pycode, Symbol
    >>> from sympy.codegen.ast import Print, stderr, QuotedString
    >>> print(pycode(Print(['x'], file=stderr)))
    print(x, file=sys.stderr)
    >>> x = Symbol('x')
    >>> print(pycode(Print([QuotedString('x')], file=stderr)))  # print literally "x"
    print("x", file=sys.stderr)

    """
    __slots__ = _fields = ('name',)
    _construct_name = String
stdout = Stream('stdout')
stderr = Stream('stderr')

class Print(Token):
    """ Represents print command in the code.

    Parameters
    ==========

    formatstring : str
    *args : Basic instances (or convertible to such through sympify)

    Examples
    ========

    >>> from sympy.codegen.ast import Print
    >>> from sympy import pycode
    >>> print(pycode(Print('x y'.split(), "coordinate: %12.5g %12.5g\\\\n")))
    print("coordinate: %12.5g %12.5g\\n" % (x, y), end="")

    """
    __slots__ = _fields = ('print_args', 'format_string', 'file')
    defaults = {'format_string': none, 'file': none}
    _construct_print_args = staticmethod(_mk_Tuple)
    _construct_format_string = QuotedString
    _construct_file = Stream

class FunctionPrototype(Node):
    """ Represents a function prototype

    Allows the user to generate forward declaration in e.g. C/C++.

    Parameters
    ==========

    return_type : Type
    name : str
    parameters: iterable of Variable instances
    attrs : iterable of Attribute instances

    Examples
    ========

    >>> from sympy import ccode, symbols
    >>> from sympy.codegen.ast import real, FunctionPrototype
    >>> x, y = symbols('x y', real=True)
    >>> fp = FunctionPrototype(real, 'foo', [x, y])
    >>> ccode(fp)
    'double foo(double x, double y)'

    """
    __slots__ = ('return_type', 'name', 'parameters')
    _fields: tuple[str, ...] = __slots__ + Node._fields
    _construct_return_type = Type
    _construct_name = String

    @staticmethod
    def _construct_parameters(args):
        if False:
            for i in range(10):
                print('nop')

        def _var(arg):
            if False:
                while True:
                    i = 10
            if isinstance(arg, Declaration):
                return arg.variable
            elif isinstance(arg, Variable):
                return arg
            else:
                return Variable.deduced(arg)
        return Tuple(*map(_var, args))

    @classmethod
    def from_FunctionDefinition(cls, func_def):
        if False:
            i = 10
            return i + 15
        if not isinstance(func_def, FunctionDefinition):
            raise TypeError('func_def is not an instance of FunctionDefinition')
        return cls(**func_def.kwargs(exclude=('body',)))

class FunctionDefinition(FunctionPrototype):
    """ Represents a function definition in the code.

    Parameters
    ==========

    return_type : Type
    name : str
    parameters: iterable of Variable instances
    body : CodeBlock or iterable
    attrs : iterable of Attribute instances

    Examples
    ========

    >>> from sympy import ccode, symbols
    >>> from sympy.codegen.ast import real, FunctionPrototype
    >>> x, y = symbols('x y', real=True)
    >>> fp = FunctionPrototype(real, 'foo', [x, y])
    >>> ccode(fp)
    'double foo(double x, double y)'
    >>> from sympy.codegen.ast import FunctionDefinition, Return
    >>> body = [Return(x*y)]
    >>> fd = FunctionDefinition.from_FunctionPrototype(fp, body)
    >>> print(ccode(fd))
    double foo(double x, double y){
        return x*y;
    }
    """
    __slots__ = ('body',)
    _fields = FunctionPrototype._fields[:-1] + __slots__ + Node._fields

    @classmethod
    def _construct_body(cls, itr):
        if False:
            print('Hello World!')
        if isinstance(itr, CodeBlock):
            return itr
        else:
            return CodeBlock(*itr)

    @classmethod
    def from_FunctionPrototype(cls, func_proto, body):
        if False:
            while True:
                i = 10
        if not isinstance(func_proto, FunctionPrototype):
            raise TypeError('func_proto is not an instance of FunctionPrototype')
        return cls(body=body, **func_proto.kwargs())

class Return(Token):
    """ Represents a return command in the code.

    Parameters
    ==========

    return : Basic

    Examples
    ========

    >>> from sympy.codegen.ast import Return
    >>> from sympy.printing.pycode import pycode
    >>> from sympy import Symbol
    >>> x = Symbol('x')
    >>> print(pycode(Return(x)))
    return x

    """
    __slots__ = _fields = ('return',)
    _construct_return = staticmethod(_sympify)

class FunctionCall(Token, Expr):
    """ Represents a call to a function in the code.

    Parameters
    ==========

    name : str
    function_args : Tuple

    Examples
    ========

    >>> from sympy.codegen.ast import FunctionCall
    >>> from sympy import pycode
    >>> fcall = FunctionCall('foo', 'bar baz'.split())
    >>> print(pycode(fcall))
    foo(bar, baz)

    """
    __slots__ = _fields = ('name', 'function_args')
    _construct_name = String
    _construct_function_args = staticmethod(lambda args: Tuple(*args))

class Raise(Token):
    """ Prints as 'raise ...' in Python, 'throw ...' in C++"""
    __slots__ = _fields = ('exception',)

class RuntimeError_(Token):
    """ Represents 'std::runtime_error' in C++ and 'RuntimeError' in Python.

    Note that the latter is uncommon, and you might want to use e.g. ValueError.
    """
    __slots__ = _fields = ('message',)
    _construct_message = String