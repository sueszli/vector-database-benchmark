from sympy.core.expr import Expr
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.matrices.dense import Matrix
from sympy.printing.pretty.stringpict import prettyForm
from sympy.core.containers import Tuple
from sympy.utilities.iterables import is_sequence
from sympy.physics.quantum.dagger import Dagger
from sympy.physics.quantum.matrixutils import numpy_ndarray, scipy_sparse_matrix, to_sympy, to_numpy, to_scipy_sparse
__all__ = ['QuantumError', 'QExpr']

class QuantumError(Exception):
    pass

def _qsympify_sequence(seq):
    if False:
        print('Hello World!')
    "Convert elements of a sequence to standard form.\n\n    This is like sympify, but it performs special logic for arguments passed\n    to QExpr. The following conversions are done:\n\n    * (list, tuple, Tuple) => _qsympify_sequence each element and convert\n      sequence to a Tuple.\n    * basestring => Symbol\n    * Matrix => Matrix\n    * other => sympify\n\n    Strings are passed to Symbol, not sympify to make sure that variables like\n    'pi' are kept as Symbols, not the SymPy built-in number subclasses.\n\n    Examples\n    ========\n\n    >>> from sympy.physics.quantum.qexpr import _qsympify_sequence\n    >>> _qsympify_sequence((1,2,[3,4,[1,]]))\n    (1, 2, (3, 4, (1,)))\n\n    "
    return tuple(__qsympify_sequence_helper(seq))

def __qsympify_sequence_helper(seq):
    if False:
        print('Hello World!')
    '\n       Helper function for _qsympify_sequence\n       This function does the actual work.\n    '
    if not is_sequence(seq):
        if isinstance(seq, Matrix):
            return seq
        elif isinstance(seq, str):
            return Symbol(seq)
        else:
            return sympify(seq)
    if isinstance(seq, QExpr):
        return seq
    result = [__qsympify_sequence_helper(item) for item in seq]
    return Tuple(*result)

class QExpr(Expr):
    """A base class for all quantum object like operators and states."""
    __slots__ = ('hilbert_space',)
    is_commutative = False
    _label_separator = ''

    @property
    def free_symbols(self):
        if False:
            print('Hello World!')
        return {self}

    def __new__(cls, *args, **kwargs):
        if False:
            return 10
        'Construct a new quantum object.\n\n        Parameters\n        ==========\n\n        args : tuple\n            The list of numbers or parameters that uniquely specify the\n            quantum object. For a state, this will be its symbol or its\n            set of quantum numbers.\n\n        Examples\n        ========\n\n        >>> from sympy.physics.quantum.qexpr import QExpr\n        >>> q = QExpr(0)\n        >>> q\n        0\n        >>> q.label\n        (0,)\n        >>> q.hilbert_space\n        H\n        >>> q.args\n        (0,)\n        >>> q.is_commutative\n        False\n        '
        args = cls._eval_args(args, **kwargs)
        if len(args) == 0:
            args = cls._eval_args(tuple(cls.default_args()), **kwargs)
        inst = Expr.__new__(cls, *args)
        inst.hilbert_space = cls._eval_hilbert_space(args)
        return inst

    @classmethod
    def _new_rawargs(cls, hilbert_space, *args, **old_assumptions):
        if False:
            for i in range(10):
                print('nop')
        'Create new instance of this class with hilbert_space and args.\n\n        This is used to bypass the more complex logic in the ``__new__``\n        method in cases where you already have the exact ``hilbert_space``\n        and ``args``. This should be used when you are positive these\n        arguments are valid, in their final, proper form and want to optimize\n        the creation of the object.\n        '
        obj = Expr.__new__(cls, *args, **old_assumptions)
        obj.hilbert_space = hilbert_space
        return obj

    @property
    def label(self):
        if False:
            return 10
        'The label is the unique set of identifiers for the object.\n\n        Usually, this will include all of the information about the state\n        *except* the time (in the case of time-dependent objects).\n\n        This must be a tuple, rather than a Tuple.\n        '
        if len(self.args) == 0:
            return self._eval_args(list(self.default_args()))
        else:
            return self.args

    @property
    def is_symbolic(self):
        if False:
            print('Hello World!')
        return True

    @classmethod
    def default_args(self):
        if False:
            for i in range(10):
                print('nop')
        'If no arguments are specified, then this will return a default set\n        of arguments to be run through the constructor.\n\n        NOTE: Any classes that override this MUST return a tuple of arguments.\n        Should be overridden by subclasses to specify the default arguments for kets and operators\n        '
        raise NotImplementedError('No default arguments for this class!')

    def _eval_adjoint(self):
        if False:
            while True:
                i = 10
        obj = Expr._eval_adjoint(self)
        if obj is None:
            obj = Expr.__new__(Dagger, self)
        if isinstance(obj, QExpr):
            obj.hilbert_space = self.hilbert_space
        return obj

    @classmethod
    def _eval_args(cls, args):
        if False:
            return 10
        'Process the args passed to the __new__ method.\n\n        This simply runs args through _qsympify_sequence.\n        '
        return _qsympify_sequence(args)

    @classmethod
    def _eval_hilbert_space(cls, args):
        if False:
            print('Hello World!')
        'Compute the Hilbert space instance from the args.\n        '
        from sympy.physics.quantum.hilbert import HilbertSpace
        return HilbertSpace()

    def _print_sequence(self, seq, sep, printer, *args):
        if False:
            for i in range(10):
                print('nop')
        result = []
        for item in seq:
            result.append(printer._print(item, *args))
        return sep.join(result)

    def _print_sequence_pretty(self, seq, sep, printer, *args):
        if False:
            return 10
        pform = printer._print(seq[0], *args)
        for item in seq[1:]:
            pform = prettyForm(*pform.right(sep))
            pform = prettyForm(*pform.right(printer._print(item, *args)))
        return pform

    def _print_subscript_pretty(self, a, b):
        if False:
            while True:
                i = 10
        top = prettyForm(*b.left(' ' * a.width()))
        bot = prettyForm(*a.right(' ' * b.width()))
        return prettyForm(*bot.below(top), binding=prettyForm.POW)

    def _print_superscript_pretty(self, a, b):
        if False:
            while True:
                i = 10
        return a ** b

    def _print_parens_pretty(self, pform, left='(', right=')'):
        if False:
            i = 10
            return i + 15
        return prettyForm(*pform.parens(left=left, right=right))

    def _print_label(self, printer, *args):
        if False:
            print('Hello World!')
        'Prints the label of the QExpr\n\n        This method prints self.label, using self._label_separator to separate\n        the elements. This method should not be overridden, instead, override\n        _print_contents to change printing behavior.\n        '
        return self._print_sequence(self.label, self._label_separator, printer, *args)

    def _print_label_repr(self, printer, *args):
        if False:
            i = 10
            return i + 15
        return self._print_sequence(self.label, ',', printer, *args)

    def _print_label_pretty(self, printer, *args):
        if False:
            while True:
                i = 10
        return self._print_sequence_pretty(self.label, self._label_separator, printer, *args)

    def _print_label_latex(self, printer, *args):
        if False:
            for i in range(10):
                print('nop')
        return self._print_sequence(self.label, self._label_separator, printer, *args)

    def _print_contents(self, printer, *args):
        if False:
            print('Hello World!')
        'Printer for contents of QExpr\n\n        Handles the printing of any unique identifying contents of a QExpr to\n        print as its contents, such as any variables or quantum numbers. The\n        default is to print the label, which is almost always the args. This\n        should not include printing of any brackets or parentheses.\n        '
        return self._print_label(printer, *args)

    def _print_contents_pretty(self, printer, *args):
        if False:
            i = 10
            return i + 15
        return self._print_label_pretty(printer, *args)

    def _print_contents_latex(self, printer, *args):
        if False:
            for i in range(10):
                print('nop')
        return self._print_label_latex(printer, *args)

    def _sympystr(self, printer, *args):
        if False:
            print('Hello World!')
        'Default printing behavior of QExpr objects\n\n        Handles the default printing of a QExpr. To add other things to the\n        printing of the object, such as an operator name to operators or\n        brackets to states, the class should override the _print/_pretty/_latex\n        functions directly and make calls to _print_contents where appropriate.\n        This allows things like InnerProduct to easily control its printing the\n        printing of contents.\n        '
        return self._print_contents(printer, *args)

    def _sympyrepr(self, printer, *args):
        if False:
            i = 10
            return i + 15
        classname = self.__class__.__name__
        label = self._print_label_repr(printer, *args)
        return '%s(%s)' % (classname, label)

    def _pretty(self, printer, *args):
        if False:
            i = 10
            return i + 15
        pform = self._print_contents_pretty(printer, *args)
        return pform

    def _latex(self, printer, *args):
        if False:
            return 10
        return self._print_contents_latex(printer, *args)

    def _represent_default_basis(self, **options):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError('This object does not have a default basis')

    def _represent(self, *, basis=None, **options):
        if False:
            print('Hello World!')
        'Represent this object in a given basis.\n\n        This method dispatches to the actual methods that perform the\n        representation. Subclases of QExpr should define various methods to\n        determine how the object will be represented in various bases. The\n        format of these methods is::\n\n            def _represent_BasisName(self, basis, **options):\n\n        Thus to define how a quantum object is represented in the basis of\n        the operator Position, you would define::\n\n            def _represent_Position(self, basis, **options):\n\n        Usually, basis object will be instances of Operator subclasses, but\n        there is a chance we will relax this in the future to accommodate other\n        types of basis sets that are not associated with an operator.\n\n        If the ``format`` option is given it can be ("sympy", "numpy",\n        "scipy.sparse"). This will ensure that any matrices that result from\n        representing the object are returned in the appropriate matrix format.\n\n        Parameters\n        ==========\n\n        basis : Operator\n            The Operator whose basis functions will be used as the basis for\n            representation.\n        options : dict\n            A dictionary of key/value pairs that give options and hints for\n            the representation, such as the number of basis functions to\n            be used.\n        '
        if basis is None:
            result = self._represent_default_basis(**options)
        else:
            result = dispatch_method(self, '_represent', basis, **options)
        format = options.get('format', 'sympy')
        result = self._format_represent(result, format)
        return result

    def _format_represent(self, result, format):
        if False:
            i = 10
            return i + 15
        if format == 'sympy' and (not isinstance(result, Matrix)):
            return to_sympy(result)
        elif format == 'numpy' and (not isinstance(result, numpy_ndarray)):
            return to_numpy(result)
        elif format == 'scipy.sparse' and (not isinstance(result, scipy_sparse_matrix)):
            return to_scipy_sparse(result)
        return result

def split_commutative_parts(e):
    if False:
        while True:
            i = 10
    'Split into commutative and non-commutative parts.'
    (c_part, nc_part) = e.args_cnc()
    c_part = list(c_part)
    return (c_part, nc_part)

def split_qexpr_parts(e):
    if False:
        print('Hello World!')
    'Split an expression into Expr and noncommutative QExpr parts.'
    expr_part = []
    qexpr_part = []
    for arg in e.args:
        if not isinstance(arg, QExpr):
            expr_part.append(arg)
        else:
            qexpr_part.append(arg)
    return (expr_part, qexpr_part)

def dispatch_method(self, basename, arg, **options):
    if False:
        print('Hello World!')
    'Dispatch a method to the proper handlers.'
    method_name = '%s_%s' % (basename, arg.__class__.__name__)
    if hasattr(self, method_name):
        f = getattr(self, method_name)
        result = f(arg, **options)
        if result is not None:
            return result
    raise NotImplementedError('%s.%s cannot handle: %r' % (self.__class__.__name__, basename, arg))