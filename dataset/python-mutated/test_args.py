"""Test whether all elements of cls.args are instances of Basic. """
import os
import re
from sympy.assumptions.ask import Q
from sympy.core.basic import Basic
from sympy.core.function import Function, Lambda
from sympy.core.numbers import Rational, oo, pi
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.elementary.exponential import exp, log
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import sin
from sympy.testing.pytest import SKIP
(a, b, c, x, y, z) = symbols('a,b,c,x,y,z')
whitelist = ['sympy.assumptions.predicates', 'sympy.assumptions.relation.equality']

def test_all_classes_are_tested():
    if False:
        i = 10
        return i + 15
    this = os.path.split(__file__)[0]
    path = os.path.join(this, os.pardir, os.pardir)
    sympy_path = os.path.abspath(path)
    prefix = os.path.split(sympy_path)[0] + os.sep
    re_cls = re.compile('^class ([A-Za-z][A-Za-z0-9_]*)\\s*\\(', re.MULTILINE)
    modules = {}
    for (root, dirs, files) in os.walk(sympy_path):
        module = root.replace(prefix, '').replace(os.sep, '.')
        for file in files:
            if file.startswith(('_', 'test_', 'bench_')):
                continue
            if not file.endswith('.py'):
                continue
            with open(os.path.join(root, file), encoding='utf-8') as f:
                text = f.read()
            submodule = module + '.' + file[:-3]
            if any((submodule.startswith(wpath) for wpath in whitelist)):
                continue
            names = re_cls.findall(text)
            if not names:
                continue
            try:
                mod = __import__(submodule, fromlist=names)
            except ImportError:
                continue

            def is_Basic(name):
                if False:
                    print('Hello World!')
                cls = getattr(mod, name)
                if hasattr(cls, '_sympy_deprecated_func'):
                    cls = cls._sympy_deprecated_func
                if not isinstance(cls, type):
                    cls = type(cls)
                return issubclass(cls, Basic)
            names = list(filter(is_Basic, names))
            if names:
                modules[submodule] = names
    ns = globals()
    failed = []
    for (module, names) in modules.items():
        mod = module.replace('.', '__')
        for name in names:
            test = 'test_' + mod + '__' + name
            if test not in ns:
                failed.append(module + '.' + name)
    assert not failed, 'Missing classes: %s.  Please add tests for these to sympy/core/tests/test_args.py.' % ', '.join(failed)

def _test_args(obj):
    if False:
        print('Hello World!')
    all_basic = all((isinstance(arg, Basic) for arg in obj.args))
    recreatable = not obj.args or obj.func(*obj.args) == obj
    return all_basic and recreatable

def test_sympy__algebras__quaternion__Quaternion():
    if False:
        return 10
    from sympy.algebras.quaternion import Quaternion
    assert _test_args(Quaternion(x, 1, 2, 3))

def test_sympy__assumptions__assume__AppliedPredicate():
    if False:
        i = 10
        return i + 15
    from sympy.assumptions.assume import AppliedPredicate, Predicate
    assert _test_args(AppliedPredicate(Predicate('test'), 2))
    assert _test_args(Q.is_true(True))

@SKIP('abstract class')
def test_sympy__assumptions__assume__Predicate():
    if False:
        i = 10
        return i + 15
    pass

def test_predicates():
    if False:
        while True:
            i = 10
    predicates = [getattr(Q, attr) for attr in Q.__class__.__dict__ if not attr.startswith('__')]
    for p in predicates:
        assert _test_args(p)

def test_sympy__assumptions__assume__UndefinedPredicate():
    if False:
        print('Hello World!')
    from sympy.assumptions.assume import Predicate
    assert _test_args(Predicate('test'))

@SKIP('abstract class')
def test_sympy__assumptions__relation__binrel__BinaryRelation():
    if False:
        return 10
    pass

def test_sympy__assumptions__relation__binrel__AppliedBinaryRelation():
    if False:
        print('Hello World!')
    assert _test_args(Q.eq(1, 2))

def test_sympy__assumptions__wrapper__AssumptionsWrapper():
    if False:
        for i in range(10):
            print('nop')
    from sympy.assumptions.wrapper import AssumptionsWrapper
    assert _test_args(AssumptionsWrapper(x, Q.positive(x)))

@SKIP('abstract Class')
def test_sympy__codegen__ast__CodegenAST():
    if False:
        print('Hello World!')
    from sympy.codegen.ast import CodegenAST
    assert _test_args(CodegenAST())

@SKIP('abstract Class')
def test_sympy__codegen__ast__AssignmentBase():
    if False:
        print('Hello World!')
    from sympy.codegen.ast import AssignmentBase
    assert _test_args(AssignmentBase(x, 1))

@SKIP('abstract Class')
def test_sympy__codegen__ast__AugmentedAssignment():
    if False:
        print('Hello World!')
    from sympy.codegen.ast import AugmentedAssignment
    assert _test_args(AugmentedAssignment(x, 1))

def test_sympy__codegen__ast__AddAugmentedAssignment():
    if False:
        return 10
    from sympy.codegen.ast import AddAugmentedAssignment
    assert _test_args(AddAugmentedAssignment(x, 1))

def test_sympy__codegen__ast__SubAugmentedAssignment():
    if False:
        i = 10
        return i + 15
    from sympy.codegen.ast import SubAugmentedAssignment
    assert _test_args(SubAugmentedAssignment(x, 1))

def test_sympy__codegen__ast__MulAugmentedAssignment():
    if False:
        i = 10
        return i + 15
    from sympy.codegen.ast import MulAugmentedAssignment
    assert _test_args(MulAugmentedAssignment(x, 1))

def test_sympy__codegen__ast__DivAugmentedAssignment():
    if False:
        return 10
    from sympy.codegen.ast import DivAugmentedAssignment
    assert _test_args(DivAugmentedAssignment(x, 1))

def test_sympy__codegen__ast__ModAugmentedAssignment():
    if False:
        for i in range(10):
            print('nop')
    from sympy.codegen.ast import ModAugmentedAssignment
    assert _test_args(ModAugmentedAssignment(x, 1))

def test_sympy__codegen__ast__CodeBlock():
    if False:
        i = 10
        return i + 15
    from sympy.codegen.ast import CodeBlock, Assignment
    assert _test_args(CodeBlock(Assignment(x, 1), Assignment(y, 2)))

def test_sympy__codegen__ast__For():
    if False:
        i = 10
        return i + 15
    from sympy.codegen.ast import For, CodeBlock, AddAugmentedAssignment
    from sympy.sets import Range
    assert _test_args(For(x, Range(10), CodeBlock(AddAugmentedAssignment(y, 1))))

def test_sympy__codegen__ast__Token():
    if False:
        i = 10
        return i + 15
    from sympy.codegen.ast import Token
    assert _test_args(Token())

def test_sympy__codegen__ast__ContinueToken():
    if False:
        for i in range(10):
            print('nop')
    from sympy.codegen.ast import ContinueToken
    assert _test_args(ContinueToken())

def test_sympy__codegen__ast__BreakToken():
    if False:
        print('Hello World!')
    from sympy.codegen.ast import BreakToken
    assert _test_args(BreakToken())

def test_sympy__codegen__ast__NoneToken():
    if False:
        print('Hello World!')
    from sympy.codegen.ast import NoneToken
    assert _test_args(NoneToken())

def test_sympy__codegen__ast__String():
    if False:
        return 10
    from sympy.codegen.ast import String
    assert _test_args(String('foobar'))

def test_sympy__codegen__ast__QuotedString():
    if False:
        while True:
            i = 10
    from sympy.codegen.ast import QuotedString
    assert _test_args(QuotedString('foobar'))

def test_sympy__codegen__ast__Comment():
    if False:
        while True:
            i = 10
    from sympy.codegen.ast import Comment
    assert _test_args(Comment('this is a comment'))

def test_sympy__codegen__ast__Node():
    if False:
        print('Hello World!')
    from sympy.codegen.ast import Node
    assert _test_args(Node())
    assert _test_args(Node(attrs={1, 2, 3}))

def test_sympy__codegen__ast__Type():
    if False:
        for i in range(10):
            print('nop')
    from sympy.codegen.ast import Type
    assert _test_args(Type('float128'))

def test_sympy__codegen__ast__IntBaseType():
    if False:
        print('Hello World!')
    from sympy.codegen.ast import IntBaseType
    assert _test_args(IntBaseType('bigint'))

def test_sympy__codegen__ast___SizedIntType():
    if False:
        print('Hello World!')
    from sympy.codegen.ast import _SizedIntType
    assert _test_args(_SizedIntType('int128', 128))

def test_sympy__codegen__ast__SignedIntType():
    if False:
        return 10
    from sympy.codegen.ast import SignedIntType
    assert _test_args(SignedIntType('int128_with_sign', 128))

def test_sympy__codegen__ast__UnsignedIntType():
    if False:
        while True:
            i = 10
    from sympy.codegen.ast import UnsignedIntType
    assert _test_args(UnsignedIntType('unt128', 128))

def test_sympy__codegen__ast__FloatBaseType():
    if False:
        for i in range(10):
            print('nop')
    from sympy.codegen.ast import FloatBaseType
    assert _test_args(FloatBaseType('positive_real'))

def test_sympy__codegen__ast__FloatType():
    if False:
        for i in range(10):
            print('nop')
    from sympy.codegen.ast import FloatType
    assert _test_args(FloatType('float242', 242, nmant=142, nexp=99))

def test_sympy__codegen__ast__ComplexBaseType():
    if False:
        i = 10
        return i + 15
    from sympy.codegen.ast import ComplexBaseType
    assert _test_args(ComplexBaseType('positive_cmplx'))

def test_sympy__codegen__ast__ComplexType():
    if False:
        while True:
            i = 10
    from sympy.codegen.ast import ComplexType
    assert _test_args(ComplexType('complex42', 42, nmant=15, nexp=5))

def test_sympy__codegen__ast__Attribute():
    if False:
        i = 10
        return i + 15
    from sympy.codegen.ast import Attribute
    assert _test_args(Attribute('noexcept'))

def test_sympy__codegen__ast__Variable():
    if False:
        print('Hello World!')
    from sympy.codegen.ast import Variable, Type, value_const
    assert _test_args(Variable(x))
    assert _test_args(Variable(y, Type('float32'), {value_const}))
    assert _test_args(Variable(z, type=Type('float64')))

def test_sympy__codegen__ast__Pointer():
    if False:
        i = 10
        return i + 15
    from sympy.codegen.ast import Pointer, Type, pointer_const
    assert _test_args(Pointer(x))
    assert _test_args(Pointer(y, type=Type('float32')))
    assert _test_args(Pointer(z, Type('float64'), {pointer_const}))

def test_sympy__codegen__ast__Declaration():
    if False:
        while True:
            i = 10
    from sympy.codegen.ast import Declaration, Variable, Type
    vx = Variable(x, type=Type('float'))
    assert _test_args(Declaration(vx))

def test_sympy__codegen__ast__While():
    if False:
        return 10
    from sympy.codegen.ast import While, AddAugmentedAssignment
    assert _test_args(While(abs(x) < 1, [AddAugmentedAssignment(x, -1)]))

def test_sympy__codegen__ast__Scope():
    if False:
        for i in range(10):
            print('nop')
    from sympy.codegen.ast import Scope, AddAugmentedAssignment
    assert _test_args(Scope([AddAugmentedAssignment(x, -1)]))

def test_sympy__codegen__ast__Stream():
    if False:
        while True:
            i = 10
    from sympy.codegen.ast import Stream
    assert _test_args(Stream('stdin'))

def test_sympy__codegen__ast__Print():
    if False:
        print('Hello World!')
    from sympy.codegen.ast import Print
    assert _test_args(Print([x, y]))
    assert _test_args(Print([x, y], '%d %d'))

def test_sympy__codegen__ast__FunctionPrototype():
    if False:
        for i in range(10):
            print('nop')
    from sympy.codegen.ast import FunctionPrototype, real, Declaration, Variable
    inp_x = Declaration(Variable(x, type=real))
    assert _test_args(FunctionPrototype(real, 'pwer', [inp_x]))

def test_sympy__codegen__ast__FunctionDefinition():
    if False:
        while True:
            i = 10
    from sympy.codegen.ast import FunctionDefinition, real, Declaration, Variable, Assignment
    inp_x = Declaration(Variable(x, type=real))
    assert _test_args(FunctionDefinition(real, 'pwer', [inp_x], [Assignment(x, x ** 2)]))

def test_sympy__codegen__ast__Raise():
    if False:
        while True:
            i = 10
    from sympy.codegen.ast import Raise
    assert _test_args(Raise(x))

def test_sympy__codegen__ast__Return():
    if False:
        for i in range(10):
            print('nop')
    from sympy.codegen.ast import Return
    assert _test_args(Return(x))

def test_sympy__codegen__ast__RuntimeError_():
    if False:
        print('Hello World!')
    from sympy.codegen.ast import RuntimeError_
    assert _test_args(RuntimeError_('"message"'))

def test_sympy__codegen__ast__FunctionCall():
    if False:
        for i in range(10):
            print('nop')
    from sympy.codegen.ast import FunctionCall
    assert _test_args(FunctionCall('pwer', [x]))

def test_sympy__codegen__ast__Element():
    if False:
        for i in range(10):
            print('nop')
    from sympy.codegen.ast import Element
    assert _test_args(Element('x', range(3)))

def test_sympy__codegen__cnodes__CommaOperator():
    if False:
        for i in range(10):
            print('nop')
    from sympy.codegen.cnodes import CommaOperator
    assert _test_args(CommaOperator(1, 2))

def test_sympy__codegen__cnodes__goto():
    if False:
        while True:
            i = 10
    from sympy.codegen.cnodes import goto
    assert _test_args(goto('early_exit'))

def test_sympy__codegen__cnodes__Label():
    if False:
        print('Hello World!')
    from sympy.codegen.cnodes import Label
    assert _test_args(Label('early_exit'))

def test_sympy__codegen__cnodes__PreDecrement():
    if False:
        print('Hello World!')
    from sympy.codegen.cnodes import PreDecrement
    assert _test_args(PreDecrement(x))

def test_sympy__codegen__cnodes__PostDecrement():
    if False:
        return 10
    from sympy.codegen.cnodes import PostDecrement
    assert _test_args(PostDecrement(x))

def test_sympy__codegen__cnodes__PreIncrement():
    if False:
        print('Hello World!')
    from sympy.codegen.cnodes import PreIncrement
    assert _test_args(PreIncrement(x))

def test_sympy__codegen__cnodes__PostIncrement():
    if False:
        for i in range(10):
            print('nop')
    from sympy.codegen.cnodes import PostIncrement
    assert _test_args(PostIncrement(x))

def test_sympy__codegen__cnodes__struct():
    if False:
        print('Hello World!')
    from sympy.codegen.ast import real, Variable
    from sympy.codegen.cnodes import struct
    assert _test_args(struct(declarations=[Variable(x, type=real), Variable(y, type=real)]))

def test_sympy__codegen__cnodes__union():
    if False:
        print('Hello World!')
    from sympy.codegen.ast import float32, int32, Variable
    from sympy.codegen.cnodes import union
    assert _test_args(union(declarations=[Variable(x, type=float32), Variable(y, type=int32)]))

def test_sympy__codegen__cxxnodes__using():
    if False:
        return 10
    from sympy.codegen.cxxnodes import using
    assert _test_args(using('std::vector'))
    assert _test_args(using('std::vector', 'vec'))

def test_sympy__codegen__fnodes__Program():
    if False:
        for i in range(10):
            print('nop')
    from sympy.codegen.fnodes import Program
    assert _test_args(Program('foobar', []))

def test_sympy__codegen__fnodes__Module():
    if False:
        i = 10
        return i + 15
    from sympy.codegen.fnodes import Module
    assert _test_args(Module('foobar', [], []))

def test_sympy__codegen__fnodes__Subroutine():
    if False:
        i = 10
        return i + 15
    from sympy.codegen.fnodes import Subroutine
    x = symbols('x', real=True)
    assert _test_args(Subroutine('foo', [x], []))

def test_sympy__codegen__fnodes__GoTo():
    if False:
        i = 10
        return i + 15
    from sympy.codegen.fnodes import GoTo
    assert _test_args(GoTo([10]))
    assert _test_args(GoTo([10, 20], x > 1))

def test_sympy__codegen__fnodes__FortranReturn():
    if False:
        for i in range(10):
            print('nop')
    from sympy.codegen.fnodes import FortranReturn
    assert _test_args(FortranReturn(10))

def test_sympy__codegen__fnodes__Extent():
    if False:
        i = 10
        return i + 15
    from sympy.codegen.fnodes import Extent
    assert _test_args(Extent())
    assert _test_args(Extent(None))
    assert _test_args(Extent(':'))
    assert _test_args(Extent(-3, 4))
    assert _test_args(Extent(x, y))

def test_sympy__codegen__fnodes__use_rename():
    if False:
        while True:
            i = 10
    from sympy.codegen.fnodes import use_rename
    assert _test_args(use_rename('loc', 'glob'))

def test_sympy__codegen__fnodes__use():
    if False:
        return 10
    from sympy.codegen.fnodes import use
    assert _test_args(use('modfoo', only='bar'))

def test_sympy__codegen__fnodes__SubroutineCall():
    if False:
        for i in range(10):
            print('nop')
    from sympy.codegen.fnodes import SubroutineCall
    assert _test_args(SubroutineCall('foo', ['bar', 'baz']))

def test_sympy__codegen__fnodes__Do():
    if False:
        while True:
            i = 10
    from sympy.codegen.fnodes import Do
    assert _test_args(Do([], 'i', 1, 42))

def test_sympy__codegen__fnodes__ImpliedDoLoop():
    if False:
        return 10
    from sympy.codegen.fnodes import ImpliedDoLoop
    assert _test_args(ImpliedDoLoop('i', 'i', 1, 42))

def test_sympy__codegen__fnodes__ArrayConstructor():
    if False:
        print('Hello World!')
    from sympy.codegen.fnodes import ArrayConstructor
    assert _test_args(ArrayConstructor([1, 2, 3]))
    from sympy.codegen.fnodes import ImpliedDoLoop
    idl = ImpliedDoLoop('i', 'i', 1, 42)
    assert _test_args(ArrayConstructor([1, idl, 3]))

def test_sympy__codegen__fnodes__sum_():
    if False:
        for i in range(10):
            print('nop')
    from sympy.codegen.fnodes import sum_
    assert _test_args(sum_('arr'))

def test_sympy__codegen__fnodes__product_():
    if False:
        print('Hello World!')
    from sympy.codegen.fnodes import product_
    assert _test_args(product_('arr'))

def test_sympy__codegen__numpy_nodes__logaddexp():
    if False:
        return 10
    from sympy.codegen.numpy_nodes import logaddexp
    assert _test_args(logaddexp(x, y))

def test_sympy__codegen__numpy_nodes__logaddexp2():
    if False:
        for i in range(10):
            print('nop')
    from sympy.codegen.numpy_nodes import logaddexp2
    assert _test_args(logaddexp2(x, y))

def test_sympy__codegen__pynodes__List():
    if False:
        return 10
    from sympy.codegen.pynodes import List
    assert _test_args(List(1, 2, 3))

def test_sympy__codegen__pynodes__NumExprEvaluate():
    if False:
        for i in range(10):
            print('nop')
    from sympy.codegen.pynodes import NumExprEvaluate
    assert _test_args(NumExprEvaluate(x))

def test_sympy__codegen__scipy_nodes__cosm1():
    if False:
        print('Hello World!')
    from sympy.codegen.scipy_nodes import cosm1
    assert _test_args(cosm1(x))

def test_sympy__codegen__scipy_nodes__powm1():
    if False:
        i = 10
        return i + 15
    from sympy.codegen.scipy_nodes import powm1
    assert _test_args(powm1(x, y))

def test_sympy__codegen__abstract_nodes__List():
    if False:
        print('Hello World!')
    from sympy.codegen.abstract_nodes import List
    assert _test_args(List(1, 2, 3))

def test_sympy__combinatorics__graycode__GrayCode():
    if False:
        print('Hello World!')
    from sympy.combinatorics.graycode import GrayCode
    assert _test_args(GrayCode(3, start='100'))
    assert _test_args(GrayCode(3, rank=1))

def test_sympy__combinatorics__permutations__Permutation():
    if False:
        while True:
            i = 10
    from sympy.combinatorics.permutations import Permutation
    assert _test_args(Permutation([0, 1, 2, 3]))

def test_sympy__combinatorics__permutations__AppliedPermutation():
    if False:
        return 10
    from sympy.combinatorics.permutations import Permutation
    from sympy.combinatorics.permutations import AppliedPermutation
    p = Permutation([0, 1, 2, 3])
    assert _test_args(AppliedPermutation(p, x))

def test_sympy__combinatorics__perm_groups__PermutationGroup():
    if False:
        for i in range(10):
            print('nop')
    from sympy.combinatorics.permutations import Permutation
    from sympy.combinatorics.perm_groups import PermutationGroup
    assert _test_args(PermutationGroup([Permutation([0, 1])]))

def test_sympy__combinatorics__polyhedron__Polyhedron():
    if False:
        print('Hello World!')
    from sympy.combinatorics.permutations import Permutation
    from sympy.combinatorics.polyhedron import Polyhedron
    from sympy.abc import w, x, y, z
    pgroup = [Permutation([[0, 1, 2], [3]]), Permutation([[0, 1, 3], [2]]), Permutation([[0, 2, 3], [1]]), Permutation([[1, 2, 3], [0]]), Permutation([[0, 1], [2, 3]]), Permutation([[0, 2], [1, 3]]), Permutation([[0, 3], [1, 2]]), Permutation([[0, 1, 2, 3]])]
    corners = [w, x, y, z]
    faces = [(w, x, y), (w, y, z), (w, z, x), (x, y, z)]
    assert _test_args(Polyhedron(corners, faces, pgroup))

def test_sympy__combinatorics__prufer__Prufer():
    if False:
        return 10
    from sympy.combinatorics.prufer import Prufer
    assert _test_args(Prufer([[0, 1], [0, 2], [0, 3]], 4))

def test_sympy__combinatorics__partitions__Partition():
    if False:
        i = 10
        return i + 15
    from sympy.combinatorics.partitions import Partition
    assert _test_args(Partition([1]))

def test_sympy__combinatorics__partitions__IntegerPartition():
    if False:
        while True:
            i = 10
    from sympy.combinatorics.partitions import IntegerPartition
    assert _test_args(IntegerPartition([1]))

def test_sympy__concrete__products__Product():
    if False:
        print('Hello World!')
    from sympy.concrete.products import Product
    assert _test_args(Product(x, (x, 0, 10)))
    assert _test_args(Product(x, (x, 0, y), (y, 0, 10)))

@SKIP('abstract Class')
def test_sympy__concrete__expr_with_limits__ExprWithLimits():
    if False:
        while True:
            i = 10
    from sympy.concrete.expr_with_limits import ExprWithLimits
    assert _test_args(ExprWithLimits(x, (x, 0, 10)))
    assert _test_args(ExprWithLimits(x * y, (x, 0, 10.0), (y, 1.0, 3)))

@SKIP('abstract Class')
def test_sympy__concrete__expr_with_limits__AddWithLimits():
    if False:
        while True:
            i = 10
    from sympy.concrete.expr_with_limits import AddWithLimits
    assert _test_args(AddWithLimits(x, (x, 0, 10)))
    assert _test_args(AddWithLimits(x * y, (x, 0, 10), (y, 1, 3)))

@SKIP('abstract Class')
def test_sympy__concrete__expr_with_intlimits__ExprWithIntLimits():
    if False:
        print('Hello World!')
    from sympy.concrete.expr_with_intlimits import ExprWithIntLimits
    assert _test_args(ExprWithIntLimits(x, (x, 0, 10)))
    assert _test_args(ExprWithIntLimits(x * y, (x, 0, 10), (y, 1, 3)))

def test_sympy__concrete__summations__Sum():
    if False:
        return 10
    from sympy.concrete.summations import Sum
    assert _test_args(Sum(x, (x, 0, 10)))
    assert _test_args(Sum(x, (x, 0, y), (y, 0, 10)))

def test_sympy__core__add__Add():
    if False:
        while True:
            i = 10
    from sympy.core.add import Add
    assert _test_args(Add(x, y, z, 2))

def test_sympy__core__basic__Atom():
    if False:
        while True:
            i = 10
    from sympy.core.basic import Atom
    assert _test_args(Atom())

def test_sympy__core__basic__Basic():
    if False:
        i = 10
        return i + 15
    from sympy.core.basic import Basic
    assert _test_args(Basic())

def test_sympy__core__containers__Dict():
    if False:
        i = 10
        return i + 15
    from sympy.core.containers import Dict
    assert _test_args(Dict({x: y, y: z}))

def test_sympy__core__containers__Tuple():
    if False:
        print('Hello World!')
    from sympy.core.containers import Tuple
    assert _test_args(Tuple(x, y, z, 2))

def test_sympy__core__expr__AtomicExpr():
    if False:
        print('Hello World!')
    from sympy.core.expr import AtomicExpr
    assert _test_args(AtomicExpr())

def test_sympy__core__expr__Expr():
    if False:
        i = 10
        return i + 15
    from sympy.core.expr import Expr
    assert _test_args(Expr())

def test_sympy__core__expr__UnevaluatedExpr():
    if False:
        print('Hello World!')
    from sympy.core.expr import UnevaluatedExpr
    from sympy.abc import x
    assert _test_args(UnevaluatedExpr(x))

def test_sympy__core__function__Application():
    if False:
        print('Hello World!')
    from sympy.core.function import Application
    assert _test_args(Application(1, 2, 3))

def test_sympy__core__function__AppliedUndef():
    if False:
        print('Hello World!')
    from sympy.core.function import AppliedUndef
    assert _test_args(AppliedUndef(1, 2, 3))

def test_sympy__core__function__Derivative():
    if False:
        return 10
    from sympy.core.function import Derivative
    assert _test_args(Derivative(2, x, y, 3))

@SKIP('abstract class')
def test_sympy__core__function__Function():
    if False:
        while True:
            i = 10
    pass

def test_sympy__core__function__Lambda():
    if False:
        while True:
            i = 10
    assert _test_args(Lambda((x, y), x + y + z))

def test_sympy__core__function__Subs():
    if False:
        for i in range(10):
            print('nop')
    from sympy.core.function import Subs
    assert _test_args(Subs(x + y, x, 2))

def test_sympy__core__function__WildFunction():
    if False:
        i = 10
        return i + 15
    from sympy.core.function import WildFunction
    assert _test_args(WildFunction('f'))

def test_sympy__core__mod__Mod():
    if False:
        return 10
    from sympy.core.mod import Mod
    assert _test_args(Mod(x, 2))

def test_sympy__core__mul__Mul():
    if False:
        print('Hello World!')
    from sympy.core.mul import Mul
    assert _test_args(Mul(2, x, y, z))

def test_sympy__core__numbers__Catalan():
    if False:
        while True:
            i = 10
    from sympy.core.numbers import Catalan
    assert _test_args(Catalan())

def test_sympy__core__numbers__ComplexInfinity():
    if False:
        while True:
            i = 10
    from sympy.core.numbers import ComplexInfinity
    assert _test_args(ComplexInfinity())

def test_sympy__core__numbers__EulerGamma():
    if False:
        i = 10
        return i + 15
    from sympy.core.numbers import EulerGamma
    assert _test_args(EulerGamma())

def test_sympy__core__numbers__Exp1():
    if False:
        print('Hello World!')
    from sympy.core.numbers import Exp1
    assert _test_args(Exp1())

def test_sympy__core__numbers__Float():
    if False:
        for i in range(10):
            print('nop')
    from sympy.core.numbers import Float
    assert _test_args(Float(1.23))

def test_sympy__core__numbers__GoldenRatio():
    if False:
        i = 10
        return i + 15
    from sympy.core.numbers import GoldenRatio
    assert _test_args(GoldenRatio())

def test_sympy__core__numbers__TribonacciConstant():
    if False:
        for i in range(10):
            print('nop')
    from sympy.core.numbers import TribonacciConstant
    assert _test_args(TribonacciConstant())

def test_sympy__core__numbers__Half():
    if False:
        return 10
    from sympy.core.numbers import Half
    assert _test_args(Half())

def test_sympy__core__numbers__ImaginaryUnit():
    if False:
        while True:
            i = 10
    from sympy.core.numbers import ImaginaryUnit
    assert _test_args(ImaginaryUnit())

def test_sympy__core__numbers__Infinity():
    if False:
        print('Hello World!')
    from sympy.core.numbers import Infinity
    assert _test_args(Infinity())

def test_sympy__core__numbers__Integer():
    if False:
        print('Hello World!')
    from sympy.core.numbers import Integer
    assert _test_args(Integer(7))

@SKIP('abstract class')
def test_sympy__core__numbers__IntegerConstant():
    if False:
        return 10
    pass

def test_sympy__core__numbers__NaN():
    if False:
        for i in range(10):
            print('nop')
    from sympy.core.numbers import NaN
    assert _test_args(NaN())

def test_sympy__core__numbers__NegativeInfinity():
    if False:
        while True:
            i = 10
    from sympy.core.numbers import NegativeInfinity
    assert _test_args(NegativeInfinity())

def test_sympy__core__numbers__NegativeOne():
    if False:
        for i in range(10):
            print('nop')
    from sympy.core.numbers import NegativeOne
    assert _test_args(NegativeOne())

def test_sympy__core__numbers__Number():
    if False:
        print('Hello World!')
    from sympy.core.numbers import Number
    assert _test_args(Number(1, 7))

def test_sympy__core__numbers__NumberSymbol():
    if False:
        return 10
    from sympy.core.numbers import NumberSymbol
    assert _test_args(NumberSymbol())

def test_sympy__core__numbers__One():
    if False:
        i = 10
        return i + 15
    from sympy.core.numbers import One
    assert _test_args(One())

def test_sympy__core__numbers__Pi():
    if False:
        print('Hello World!')
    from sympy.core.numbers import Pi
    assert _test_args(Pi())

def test_sympy__core__numbers__Rational():
    if False:
        print('Hello World!')
    from sympy.core.numbers import Rational
    assert _test_args(Rational(1, 7))

@SKIP('abstract class')
def test_sympy__core__numbers__RationalConstant():
    if False:
        return 10
    pass

def test_sympy__core__numbers__Zero():
    if False:
        while True:
            i = 10
    from sympy.core.numbers import Zero
    assert _test_args(Zero())

@SKIP('abstract class')
def test_sympy__core__operations__AssocOp():
    if False:
        return 10
    pass

@SKIP('abstract class')
def test_sympy__core__operations__LatticeOp():
    if False:
        i = 10
        return i + 15
    pass

def test_sympy__core__power__Pow():
    if False:
        print('Hello World!')
    from sympy.core.power import Pow
    assert _test_args(Pow(x, 2))

def test_sympy__core__relational__Equality():
    if False:
        i = 10
        return i + 15
    from sympy.core.relational import Equality
    assert _test_args(Equality(x, 2))

def test_sympy__core__relational__GreaterThan():
    if False:
        i = 10
        return i + 15
    from sympy.core.relational import GreaterThan
    assert _test_args(GreaterThan(x, 2))

def test_sympy__core__relational__LessThan():
    if False:
        return 10
    from sympy.core.relational import LessThan
    assert _test_args(LessThan(x, 2))

@SKIP('abstract class')
def test_sympy__core__relational__Relational():
    if False:
        return 10
    pass

def test_sympy__core__relational__StrictGreaterThan():
    if False:
        i = 10
        return i + 15
    from sympy.core.relational import StrictGreaterThan
    assert _test_args(StrictGreaterThan(x, 2))

def test_sympy__core__relational__StrictLessThan():
    if False:
        while True:
            i = 10
    from sympy.core.relational import StrictLessThan
    assert _test_args(StrictLessThan(x, 2))

def test_sympy__core__relational__Unequality():
    if False:
        for i in range(10):
            print('nop')
    from sympy.core.relational import Unequality
    assert _test_args(Unequality(x, 2))

def test_sympy__sandbox__indexed_integrals__IndexedIntegral():
    if False:
        return 10
    from sympy.tensor import IndexedBase, Idx
    from sympy.sandbox.indexed_integrals import IndexedIntegral
    A = IndexedBase('A')
    (i, j) = symbols('i j', integer=True)
    (a1, a2) = symbols('a1:3', cls=Idx)
    assert _test_args(IndexedIntegral(A[a1], A[a2]))
    assert _test_args(IndexedIntegral(A[i], A[j]))

def test_sympy__calculus__accumulationbounds__AccumulationBounds():
    if False:
        i = 10
        return i + 15
    from sympy.calculus.accumulationbounds import AccumulationBounds
    assert _test_args(AccumulationBounds(0, 1))

def test_sympy__sets__ordinals__OmegaPower():
    if False:
        return 10
    from sympy.sets.ordinals import OmegaPower
    assert _test_args(OmegaPower(1, 1))

def test_sympy__sets__ordinals__Ordinal():
    if False:
        return 10
    from sympy.sets.ordinals import Ordinal, OmegaPower
    assert _test_args(Ordinal(OmegaPower(2, 1)))

def test_sympy__sets__ordinals__OrdinalOmega():
    if False:
        return 10
    from sympy.sets.ordinals import OrdinalOmega
    assert _test_args(OrdinalOmega())

def test_sympy__sets__ordinals__OrdinalZero():
    if False:
        print('Hello World!')
    from sympy.sets.ordinals import OrdinalZero
    assert _test_args(OrdinalZero())

def test_sympy__sets__powerset__PowerSet():
    if False:
        return 10
    from sympy.sets.powerset import PowerSet
    from sympy.core.singleton import S
    assert _test_args(PowerSet(S.EmptySet))

def test_sympy__sets__sets__EmptySet():
    if False:
        while True:
            i = 10
    from sympy.sets.sets import EmptySet
    assert _test_args(EmptySet())

def test_sympy__sets__sets__UniversalSet():
    if False:
        return 10
    from sympy.sets.sets import UniversalSet
    assert _test_args(UniversalSet())

def test_sympy__sets__sets__FiniteSet():
    if False:
        while True:
            i = 10
    from sympy.sets.sets import FiniteSet
    assert _test_args(FiniteSet(x, y, z))

def test_sympy__sets__sets__Interval():
    if False:
        while True:
            i = 10
    from sympy.sets.sets import Interval
    assert _test_args(Interval(0, 1))

def test_sympy__sets__sets__ProductSet():
    if False:
        for i in range(10):
            print('nop')
    from sympy.sets.sets import ProductSet, Interval
    assert _test_args(ProductSet(Interval(0, 1), Interval(0, 1)))

@SKIP('does it make sense to test this?')
def test_sympy__sets__sets__Set():
    if False:
        for i in range(10):
            print('nop')
    from sympy.sets.sets import Set
    assert _test_args(Set())

def test_sympy__sets__sets__Intersection():
    if False:
        i = 10
        return i + 15
    from sympy.sets.sets import Intersection, Interval
    from sympy.core.symbol import Symbol
    x = Symbol('x')
    y = Symbol('y')
    S = Intersection(Interval(0, x), Interval(y, 1))
    assert isinstance(S, Intersection)
    assert _test_args(S)

def test_sympy__sets__sets__Union():
    if False:
        return 10
    from sympy.sets.sets import Union, Interval
    assert _test_args(Union(Interval(0, 1), Interval(2, 3)))

def test_sympy__sets__sets__Complement():
    if False:
        return 10
    from sympy.sets.sets import Complement, Interval
    assert _test_args(Complement(Interval(0, 2), Interval(0, 1)))

def test_sympy__sets__sets__SymmetricDifference():
    if False:
        while True:
            i = 10
    from sympy.sets.sets import FiniteSet, SymmetricDifference
    assert _test_args(SymmetricDifference(FiniteSet(1, 2, 3), FiniteSet(2, 3, 4)))

def test_sympy__sets__sets__DisjointUnion():
    if False:
        while True:
            i = 10
    from sympy.sets.sets import FiniteSet, DisjointUnion
    assert _test_args(DisjointUnion(FiniteSet(1, 2, 3), FiniteSet(2, 3, 4)))

def test_sympy__physics__quantum__trace__Tr():
    if False:
        for i in range(10):
            print('nop')
    from sympy.physics.quantum.trace import Tr
    (a, b) = symbols('a b', commutative=False)
    assert _test_args(Tr(a + b))

def test_sympy__sets__setexpr__SetExpr():
    if False:
        return 10
    from sympy.sets.setexpr import SetExpr
    from sympy.sets.sets import Interval
    assert _test_args(SetExpr(Interval(0, 1)))

def test_sympy__sets__fancysets__Rationals():
    if False:
        return 10
    from sympy.sets.fancysets import Rationals
    assert _test_args(Rationals())

def test_sympy__sets__fancysets__Naturals():
    if False:
        print('Hello World!')
    from sympy.sets.fancysets import Naturals
    assert _test_args(Naturals())

def test_sympy__sets__fancysets__Naturals0():
    if False:
        while True:
            i = 10
    from sympy.sets.fancysets import Naturals0
    assert _test_args(Naturals0())

def test_sympy__sets__fancysets__Integers():
    if False:
        i = 10
        return i + 15
    from sympy.sets.fancysets import Integers
    assert _test_args(Integers())

def test_sympy__sets__fancysets__Reals():
    if False:
        for i in range(10):
            print('nop')
    from sympy.sets.fancysets import Reals
    assert _test_args(Reals())

def test_sympy__sets__fancysets__Complexes():
    if False:
        return 10
    from sympy.sets.fancysets import Complexes
    assert _test_args(Complexes())

def test_sympy__sets__fancysets__ComplexRegion():
    if False:
        for i in range(10):
            print('nop')
    from sympy.sets.fancysets import ComplexRegion
    from sympy.core.singleton import S
    from sympy.sets import Interval
    a = Interval(0, 1)
    b = Interval(2, 3)
    theta = Interval(0, 2 * S.Pi)
    assert _test_args(ComplexRegion(a * b))
    assert _test_args(ComplexRegion(a * theta, polar=True))

def test_sympy__sets__fancysets__CartesianComplexRegion():
    if False:
        i = 10
        return i + 15
    from sympy.sets.fancysets import CartesianComplexRegion
    from sympy.sets import Interval
    a = Interval(0, 1)
    b = Interval(2, 3)
    assert _test_args(CartesianComplexRegion(a * b))

def test_sympy__sets__fancysets__PolarComplexRegion():
    if False:
        while True:
            i = 10
    from sympy.sets.fancysets import PolarComplexRegion
    from sympy.core.singleton import S
    from sympy.sets import Interval
    a = Interval(0, 1)
    theta = Interval(0, 2 * S.Pi)
    assert _test_args(PolarComplexRegion(a * theta))

def test_sympy__sets__fancysets__ImageSet():
    if False:
        return 10
    from sympy.sets.fancysets import ImageSet
    from sympy.core.singleton import S
    from sympy.core.symbol import Symbol
    x = Symbol('x')
    assert _test_args(ImageSet(Lambda(x, x ** 2), S.Naturals))

def test_sympy__sets__fancysets__Range():
    if False:
        while True:
            i = 10
    from sympy.sets.fancysets import Range
    assert _test_args(Range(1, 5, 1))

def test_sympy__sets__conditionset__ConditionSet():
    if False:
        while True:
            i = 10
    from sympy.sets.conditionset import ConditionSet
    from sympy.core.singleton import S
    from sympy.core.symbol import Symbol
    x = Symbol('x')
    assert _test_args(ConditionSet(x, Eq(x ** 2, 1), S.Reals))

def test_sympy__sets__contains__Contains():
    if False:
        for i in range(10):
            print('nop')
    from sympy.sets.fancysets import Range
    from sympy.sets.contains import Contains
    assert _test_args(Contains(x, Range(0, 10, 2)))
from sympy.stats.crv_types import NormalDistribution
nd = NormalDistribution(0, 1)
from sympy.stats.frv_types import DieDistribution
die = DieDistribution(6)

def test_sympy__stats__crv__ContinuousDomain():
    if False:
        for i in range(10):
            print('nop')
    from sympy.sets.sets import Interval
    from sympy.stats.crv import ContinuousDomain
    assert _test_args(ContinuousDomain({x}, Interval(-oo, oo)))

def test_sympy__stats__crv__SingleContinuousDomain():
    if False:
        for i in range(10):
            print('nop')
    from sympy.sets.sets import Interval
    from sympy.stats.crv import SingleContinuousDomain
    assert _test_args(SingleContinuousDomain(x, Interval(-oo, oo)))

def test_sympy__stats__crv__ProductContinuousDomain():
    if False:
        for i in range(10):
            print('nop')
    from sympy.sets.sets import Interval
    from sympy.stats.crv import SingleContinuousDomain, ProductContinuousDomain
    D = SingleContinuousDomain(x, Interval(-oo, oo))
    E = SingleContinuousDomain(y, Interval(0, oo))
    assert _test_args(ProductContinuousDomain(D, E))

def test_sympy__stats__crv__ConditionalContinuousDomain():
    if False:
        i = 10
        return i + 15
    from sympy.sets.sets import Interval
    from sympy.stats.crv import SingleContinuousDomain, ConditionalContinuousDomain
    D = SingleContinuousDomain(x, Interval(-oo, oo))
    assert _test_args(ConditionalContinuousDomain(D, x > 0))

def test_sympy__stats__crv__ContinuousPSpace():
    if False:
        i = 10
        return i + 15
    from sympy.sets.sets import Interval
    from sympy.stats.crv import ContinuousPSpace, SingleContinuousDomain
    D = SingleContinuousDomain(x, Interval(-oo, oo))
    assert _test_args(ContinuousPSpace(D, nd))

def test_sympy__stats__crv__SingleContinuousPSpace():
    if False:
        print('Hello World!')
    from sympy.stats.crv import SingleContinuousPSpace
    assert _test_args(SingleContinuousPSpace(x, nd))

@SKIP('abstract class')
def test_sympy__stats__rv__Distribution():
    if False:
        for i in range(10):
            print('nop')
    pass

@SKIP('abstract class')
def test_sympy__stats__crv__SingleContinuousDistribution():
    if False:
        while True:
            i = 10
    pass

def test_sympy__stats__drv__SingleDiscreteDomain():
    if False:
        print('Hello World!')
    from sympy.stats.drv import SingleDiscreteDomain
    assert _test_args(SingleDiscreteDomain(x, S.Naturals))

def test_sympy__stats__drv__ProductDiscreteDomain():
    if False:
        print('Hello World!')
    from sympy.stats.drv import SingleDiscreteDomain, ProductDiscreteDomain
    X = SingleDiscreteDomain(x, S.Naturals)
    Y = SingleDiscreteDomain(y, S.Integers)
    assert _test_args(ProductDiscreteDomain(X, Y))

def test_sympy__stats__drv__SingleDiscretePSpace():
    if False:
        i = 10
        return i + 15
    from sympy.stats.drv import SingleDiscretePSpace
    from sympy.stats.drv_types import PoissonDistribution
    assert _test_args(SingleDiscretePSpace(x, PoissonDistribution(1)))

def test_sympy__stats__drv__DiscretePSpace():
    if False:
        return 10
    from sympy.stats.drv import DiscretePSpace, SingleDiscreteDomain
    density = Lambda(x, 2 ** (-x))
    domain = SingleDiscreteDomain(x, S.Naturals)
    assert _test_args(DiscretePSpace(domain, density))

def test_sympy__stats__drv__ConditionalDiscreteDomain():
    if False:
        print('Hello World!')
    from sympy.stats.drv import ConditionalDiscreteDomain, SingleDiscreteDomain
    X = SingleDiscreteDomain(x, S.Naturals0)
    assert _test_args(ConditionalDiscreteDomain(X, x > 2))

def test_sympy__stats__joint_rv__JointPSpace():
    if False:
        for i in range(10):
            print('nop')
    from sympy.stats.joint_rv import JointPSpace, JointDistribution
    assert _test_args(JointPSpace('X', JointDistribution(1)))

def test_sympy__stats__joint_rv__JointRandomSymbol():
    if False:
        return 10
    from sympy.stats.joint_rv import JointRandomSymbol
    assert _test_args(JointRandomSymbol(x))

def test_sympy__stats__joint_rv_types__JointDistributionHandmade():
    if False:
        return 10
    from sympy.tensor.indexed import Indexed
    from sympy.stats.joint_rv_types import JointDistributionHandmade
    (x1, x2) = (Indexed('x', i) for i in (1, 2))
    assert _test_args(JointDistributionHandmade(x1 + x2, S.Reals ** 2))

def test_sympy__stats__joint_rv__MarginalDistribution():
    if False:
        i = 10
        return i + 15
    from sympy.stats.rv import RandomSymbol
    from sympy.stats.joint_rv import MarginalDistribution
    r = RandomSymbol(S('r'))
    assert _test_args(MarginalDistribution(r, (r,)))

def test_sympy__stats__compound_rv__CompoundDistribution():
    if False:
        i = 10
        return i + 15
    from sympy.stats.compound_rv import CompoundDistribution
    from sympy.stats.drv_types import PoissonDistribution, Poisson
    r = Poisson('r', 10)
    assert _test_args(CompoundDistribution(PoissonDistribution(r)))

def test_sympy__stats__compound_rv__CompoundPSpace():
    if False:
        i = 10
        return i + 15
    from sympy.stats.compound_rv import CompoundPSpace, CompoundDistribution
    from sympy.stats.drv_types import PoissonDistribution, Poisson
    r = Poisson('r', 5)
    C = CompoundDistribution(PoissonDistribution(r))
    assert _test_args(CompoundPSpace('C', C))

@SKIP('abstract class')
def test_sympy__stats__drv__SingleDiscreteDistribution():
    if False:
        while True:
            i = 10
    pass

@SKIP('abstract class')
def test_sympy__stats__drv__DiscreteDistribution():
    if False:
        i = 10
        return i + 15
    pass

@SKIP('abstract class')
def test_sympy__stats__drv__DiscreteDomain():
    if False:
        i = 10
        return i + 15
    pass

def test_sympy__stats__rv__RandomDomain():
    if False:
        return 10
    from sympy.stats.rv import RandomDomain
    from sympy.sets.sets import FiniteSet
    assert _test_args(RandomDomain(FiniteSet(x), FiniteSet(1, 2, 3)))

def test_sympy__stats__rv__SingleDomain():
    if False:
        for i in range(10):
            print('nop')
    from sympy.stats.rv import SingleDomain
    from sympy.sets.sets import FiniteSet
    assert _test_args(SingleDomain(x, FiniteSet(1, 2, 3)))

def test_sympy__stats__rv__ConditionalDomain():
    if False:
        while True:
            i = 10
    from sympy.stats.rv import ConditionalDomain, RandomDomain
    from sympy.sets.sets import FiniteSet
    D = RandomDomain(FiniteSet(x), FiniteSet(1, 2))
    assert _test_args(ConditionalDomain(D, x > 1))

def test_sympy__stats__rv__MatrixDomain():
    if False:
        for i in range(10):
            print('nop')
    from sympy.stats.rv import MatrixDomain
    from sympy.matrices import MatrixSet
    from sympy.core.singleton import S
    assert _test_args(MatrixDomain(x, MatrixSet(2, 2, S.Reals)))

def test_sympy__stats__rv__PSpace():
    if False:
        return 10
    from sympy.stats.rv import PSpace, RandomDomain
    from sympy.sets.sets import FiniteSet
    D = RandomDomain(FiniteSet(x), FiniteSet(1, 2, 3, 4, 5, 6))
    assert _test_args(PSpace(D, die))

@SKIP('abstract Class')
def test_sympy__stats__rv__SinglePSpace():
    if False:
        for i in range(10):
            print('nop')
    pass

def test_sympy__stats__rv__RandomSymbol():
    if False:
        i = 10
        return i + 15
    from sympy.stats.rv import RandomSymbol
    from sympy.stats.crv import SingleContinuousPSpace
    A = SingleContinuousPSpace(x, nd)
    assert _test_args(RandomSymbol(x, A))

@SKIP('abstract Class')
def test_sympy__stats__rv__ProductPSpace():
    if False:
        print('Hello World!')
    pass

def test_sympy__stats__rv__IndependentProductPSpace():
    if False:
        for i in range(10):
            print('nop')
    from sympy.stats.rv import IndependentProductPSpace
    from sympy.stats.crv import SingleContinuousPSpace
    A = SingleContinuousPSpace(x, nd)
    B = SingleContinuousPSpace(y, nd)
    assert _test_args(IndependentProductPSpace(A, B))

def test_sympy__stats__rv__ProductDomain():
    if False:
        i = 10
        return i + 15
    from sympy.sets.sets import Interval
    from sympy.stats.rv import ProductDomain, SingleDomain
    D = SingleDomain(x, Interval(-oo, oo))
    E = SingleDomain(y, Interval(0, oo))
    assert _test_args(ProductDomain(D, E))

def test_sympy__stats__symbolic_probability__Probability():
    if False:
        return 10
    from sympy.stats.symbolic_probability import Probability
    from sympy.stats import Normal
    X = Normal('X', 0, 1)
    assert _test_args(Probability(X > 0))

def test_sympy__stats__symbolic_probability__Expectation():
    if False:
        return 10
    from sympy.stats.symbolic_probability import Expectation
    from sympy.stats import Normal
    X = Normal('X', 0, 1)
    assert _test_args(Expectation(X > 0))

def test_sympy__stats__symbolic_probability__Covariance():
    if False:
        for i in range(10):
            print('nop')
    from sympy.stats.symbolic_probability import Covariance
    from sympy.stats import Normal
    X = Normal('X', 0, 1)
    Y = Normal('Y', 0, 3)
    assert _test_args(Covariance(X, Y))

def test_sympy__stats__symbolic_probability__Variance():
    if False:
        print('Hello World!')
    from sympy.stats.symbolic_probability import Variance
    from sympy.stats import Normal
    X = Normal('X', 0, 1)
    assert _test_args(Variance(X))

def test_sympy__stats__symbolic_probability__Moment():
    if False:
        while True:
            i = 10
    from sympy.stats.symbolic_probability import Moment
    from sympy.stats import Normal
    X = Normal('X', 0, 1)
    assert _test_args(Moment(X, 3, 2, X > 3))

def test_sympy__stats__symbolic_probability__CentralMoment():
    if False:
        while True:
            i = 10
    from sympy.stats.symbolic_probability import CentralMoment
    from sympy.stats import Normal
    X = Normal('X', 0, 1)
    assert _test_args(CentralMoment(X, 2, X > 1))

def test_sympy__stats__frv_types__DiscreteUniformDistribution():
    if False:
        i = 10
        return i + 15
    from sympy.stats.frv_types import DiscreteUniformDistribution
    from sympy.core.containers import Tuple
    assert _test_args(DiscreteUniformDistribution(Tuple(*list(range(6)))))

def test_sympy__stats__frv_types__DieDistribution():
    if False:
        for i in range(10):
            print('nop')
    assert _test_args(die)

def test_sympy__stats__frv_types__BernoulliDistribution():
    if False:
        while True:
            i = 10
    from sympy.stats.frv_types import BernoulliDistribution
    assert _test_args(BernoulliDistribution(S.Half, 0, 1))

def test_sympy__stats__frv_types__BinomialDistribution():
    if False:
        for i in range(10):
            print('nop')
    from sympy.stats.frv_types import BinomialDistribution
    assert _test_args(BinomialDistribution(5, S.Half, 1, 0))

def test_sympy__stats__frv_types__BetaBinomialDistribution():
    if False:
        for i in range(10):
            print('nop')
    from sympy.stats.frv_types import BetaBinomialDistribution
    assert _test_args(BetaBinomialDistribution(5, 1, 1))

def test_sympy__stats__frv_types__HypergeometricDistribution():
    if False:
        return 10
    from sympy.stats.frv_types import HypergeometricDistribution
    assert _test_args(HypergeometricDistribution(10, 5, 3))

def test_sympy__stats__frv_types__RademacherDistribution():
    if False:
        while True:
            i = 10
    from sympy.stats.frv_types import RademacherDistribution
    assert _test_args(RademacherDistribution())

def test_sympy__stats__frv_types__IdealSolitonDistribution():
    if False:
        return 10
    from sympy.stats.frv_types import IdealSolitonDistribution
    assert _test_args(IdealSolitonDistribution(10))

def test_sympy__stats__frv_types__RobustSolitonDistribution():
    if False:
        return 10
    from sympy.stats.frv_types import RobustSolitonDistribution
    assert _test_args(RobustSolitonDistribution(1000, 0.5, 0.1))

def test_sympy__stats__frv__FiniteDomain():
    if False:
        return 10
    from sympy.stats.frv import FiniteDomain
    assert _test_args(FiniteDomain({(x, 1), (x, 2)}))

def test_sympy__stats__frv__SingleFiniteDomain():
    if False:
        for i in range(10):
            print('nop')
    from sympy.stats.frv import SingleFiniteDomain
    assert _test_args(SingleFiniteDomain(x, {1, 2}))

def test_sympy__stats__frv__ProductFiniteDomain():
    if False:
        for i in range(10):
            print('nop')
    from sympy.stats.frv import SingleFiniteDomain, ProductFiniteDomain
    xd = SingleFiniteDomain(x, {1, 2})
    yd = SingleFiniteDomain(y, {1, 2})
    assert _test_args(ProductFiniteDomain(xd, yd))

def test_sympy__stats__frv__ConditionalFiniteDomain():
    if False:
        print('Hello World!')
    from sympy.stats.frv import SingleFiniteDomain, ConditionalFiniteDomain
    xd = SingleFiniteDomain(x, {1, 2})
    assert _test_args(ConditionalFiniteDomain(xd, x > 1))

def test_sympy__stats__frv__FinitePSpace():
    if False:
        return 10
    from sympy.stats.frv import FinitePSpace, SingleFiniteDomain
    xd = SingleFiniteDomain(x, {1, 2, 3, 4, 5, 6})
    assert _test_args(FinitePSpace(xd, {(x, 1): S.Half, (x, 2): S.Half}))
    xd = SingleFiniteDomain(x, {1, 2})
    assert _test_args(FinitePSpace(xd, {(x, 1): S.Half, (x, 2): S.Half}))

def test_sympy__stats__frv__SingleFinitePSpace():
    if False:
        i = 10
        return i + 15
    from sympy.stats.frv import SingleFinitePSpace
    from sympy.core.symbol import Symbol
    assert _test_args(SingleFinitePSpace(Symbol('x'), die))

def test_sympy__stats__frv__ProductFinitePSpace():
    if False:
        return 10
    from sympy.stats.frv import SingleFinitePSpace, ProductFinitePSpace
    from sympy.core.symbol import Symbol
    xp = SingleFinitePSpace(Symbol('x'), die)
    yp = SingleFinitePSpace(Symbol('y'), die)
    assert _test_args(ProductFinitePSpace(xp, yp))

@SKIP('abstract class')
def test_sympy__stats__frv__SingleFiniteDistribution():
    if False:
        print('Hello World!')
    pass

@SKIP('abstract class')
def test_sympy__stats__crv__ContinuousDistribution():
    if False:
        print('Hello World!')
    pass

def test_sympy__stats__frv_types__FiniteDistributionHandmade():
    if False:
        print('Hello World!')
    from sympy.stats.frv_types import FiniteDistributionHandmade
    from sympy.core.containers import Dict
    assert _test_args(FiniteDistributionHandmade(Dict({1: 1})))

def test_sympy__stats__crv_types__ContinuousDistributionHandmade():
    if False:
        i = 10
        return i + 15
    from sympy.stats.crv_types import ContinuousDistributionHandmade
    from sympy.core.function import Lambda
    from sympy.sets.sets import Interval
    from sympy.abc import x
    assert _test_args(ContinuousDistributionHandmade(Lambda(x, 2 * x), Interval(0, 1)))

def test_sympy__stats__drv_types__DiscreteDistributionHandmade():
    if False:
        for i in range(10):
            print('nop')
    from sympy.stats.drv_types import DiscreteDistributionHandmade
    from sympy.core.function import Lambda
    from sympy.sets.sets import FiniteSet
    from sympy.abc import x
    assert _test_args(DiscreteDistributionHandmade(Lambda(x, Rational(1, 10)), FiniteSet(*range(10))))

def test_sympy__stats__rv__Density():
    if False:
        for i in range(10):
            print('nop')
    from sympy.stats.rv import Density
    from sympy.stats.crv_types import Normal
    assert _test_args(Density(Normal('x', 0, 1)))

def test_sympy__stats__crv_types__ArcsinDistribution():
    if False:
        return 10
    from sympy.stats.crv_types import ArcsinDistribution
    assert _test_args(ArcsinDistribution(0, 1))

def test_sympy__stats__crv_types__BeniniDistribution():
    if False:
        print('Hello World!')
    from sympy.stats.crv_types import BeniniDistribution
    assert _test_args(BeniniDistribution(1, 1, 1))

def test_sympy__stats__crv_types__BetaDistribution():
    if False:
        return 10
    from sympy.stats.crv_types import BetaDistribution
    assert _test_args(BetaDistribution(1, 1))

def test_sympy__stats__crv_types__BetaNoncentralDistribution():
    if False:
        i = 10
        return i + 15
    from sympy.stats.crv_types import BetaNoncentralDistribution
    assert _test_args(BetaNoncentralDistribution(1, 1, 1))

def test_sympy__stats__crv_types__BetaPrimeDistribution():
    if False:
        return 10
    from sympy.stats.crv_types import BetaPrimeDistribution
    assert _test_args(BetaPrimeDistribution(1, 1))

def test_sympy__stats__crv_types__BoundedParetoDistribution():
    if False:
        while True:
            i = 10
    from sympy.stats.crv_types import BoundedParetoDistribution
    assert _test_args(BoundedParetoDistribution(1, 1, 2))

def test_sympy__stats__crv_types__CauchyDistribution():
    if False:
        while True:
            i = 10
    from sympy.stats.crv_types import CauchyDistribution
    assert _test_args(CauchyDistribution(0, 1))

def test_sympy__stats__crv_types__ChiDistribution():
    if False:
        return 10
    from sympy.stats.crv_types import ChiDistribution
    assert _test_args(ChiDistribution(1))

def test_sympy__stats__crv_types__ChiNoncentralDistribution():
    if False:
        for i in range(10):
            print('nop')
    from sympy.stats.crv_types import ChiNoncentralDistribution
    assert _test_args(ChiNoncentralDistribution(1, 1))

def test_sympy__stats__crv_types__ChiSquaredDistribution():
    if False:
        print('Hello World!')
    from sympy.stats.crv_types import ChiSquaredDistribution
    assert _test_args(ChiSquaredDistribution(1))

def test_sympy__stats__crv_types__DagumDistribution():
    if False:
        print('Hello World!')
    from sympy.stats.crv_types import DagumDistribution
    assert _test_args(DagumDistribution(1, 1, 1))

def test_sympy__stats__crv_types__DavisDistribution():
    if False:
        return 10
    from sympy.stats.crv_types import DavisDistribution
    assert _test_args(DavisDistribution(1, 1, 1))

def test_sympy__stats__crv_types__ExGaussianDistribution():
    if False:
        print('Hello World!')
    from sympy.stats.crv_types import ExGaussianDistribution
    assert _test_args(ExGaussianDistribution(1, 1, 1))

def test_sympy__stats__crv_types__ExponentialDistribution():
    if False:
        for i in range(10):
            print('nop')
    from sympy.stats.crv_types import ExponentialDistribution
    assert _test_args(ExponentialDistribution(1))

def test_sympy__stats__crv_types__ExponentialPowerDistribution():
    if False:
        i = 10
        return i + 15
    from sympy.stats.crv_types import ExponentialPowerDistribution
    assert _test_args(ExponentialPowerDistribution(0, 1, 1))

def test_sympy__stats__crv_types__FDistributionDistribution():
    if False:
        i = 10
        return i + 15
    from sympy.stats.crv_types import FDistributionDistribution
    assert _test_args(FDistributionDistribution(1, 1))

def test_sympy__stats__crv_types__FisherZDistribution():
    if False:
        i = 10
        return i + 15
    from sympy.stats.crv_types import FisherZDistribution
    assert _test_args(FisherZDistribution(1, 1))

def test_sympy__stats__crv_types__FrechetDistribution():
    if False:
        return 10
    from sympy.stats.crv_types import FrechetDistribution
    assert _test_args(FrechetDistribution(1, 1, 1))

def test_sympy__stats__crv_types__GammaInverseDistribution():
    if False:
        while True:
            i = 10
    from sympy.stats.crv_types import GammaInverseDistribution
    assert _test_args(GammaInverseDistribution(1, 1))

def test_sympy__stats__crv_types__GammaDistribution():
    if False:
        for i in range(10):
            print('nop')
    from sympy.stats.crv_types import GammaDistribution
    assert _test_args(GammaDistribution(1, 1))

def test_sympy__stats__crv_types__GumbelDistribution():
    if False:
        return 10
    from sympy.stats.crv_types import GumbelDistribution
    assert _test_args(GumbelDistribution(1, 1, False))

def test_sympy__stats__crv_types__GompertzDistribution():
    if False:
        print('Hello World!')
    from sympy.stats.crv_types import GompertzDistribution
    assert _test_args(GompertzDistribution(1, 1))

def test_sympy__stats__crv_types__KumaraswamyDistribution():
    if False:
        print('Hello World!')
    from sympy.stats.crv_types import KumaraswamyDistribution
    assert _test_args(KumaraswamyDistribution(1, 1))

def test_sympy__stats__crv_types__LaplaceDistribution():
    if False:
        print('Hello World!')
    from sympy.stats.crv_types import LaplaceDistribution
    assert _test_args(LaplaceDistribution(0, 1))

def test_sympy__stats__crv_types__LevyDistribution():
    if False:
        i = 10
        return i + 15
    from sympy.stats.crv_types import LevyDistribution
    assert _test_args(LevyDistribution(0, 1))

def test_sympy__stats__crv_types__LogCauchyDistribution():
    if False:
        for i in range(10):
            print('nop')
    from sympy.stats.crv_types import LogCauchyDistribution
    assert _test_args(LogCauchyDistribution(0, 1))

def test_sympy__stats__crv_types__LogisticDistribution():
    if False:
        while True:
            i = 10
    from sympy.stats.crv_types import LogisticDistribution
    assert _test_args(LogisticDistribution(0, 1))

def test_sympy__stats__crv_types__LogLogisticDistribution():
    if False:
        while True:
            i = 10
    from sympy.stats.crv_types import LogLogisticDistribution
    assert _test_args(LogLogisticDistribution(1, 1))

def test_sympy__stats__crv_types__LogitNormalDistribution():
    if False:
        print('Hello World!')
    from sympy.stats.crv_types import LogitNormalDistribution
    assert _test_args(LogitNormalDistribution(0, 1))

def test_sympy__stats__crv_types__LogNormalDistribution():
    if False:
        for i in range(10):
            print('nop')
    from sympy.stats.crv_types import LogNormalDistribution
    assert _test_args(LogNormalDistribution(0, 1))

def test_sympy__stats__crv_types__LomaxDistribution():
    if False:
        i = 10
        return i + 15
    from sympy.stats.crv_types import LomaxDistribution
    assert _test_args(LomaxDistribution(1, 2))

def test_sympy__stats__crv_types__MaxwellDistribution():
    if False:
        i = 10
        return i + 15
    from sympy.stats.crv_types import MaxwellDistribution
    assert _test_args(MaxwellDistribution(1))

def test_sympy__stats__crv_types__MoyalDistribution():
    if False:
        for i in range(10):
            print('nop')
    from sympy.stats.crv_types import MoyalDistribution
    assert _test_args(MoyalDistribution(1, 2))

def test_sympy__stats__crv_types__NakagamiDistribution():
    if False:
        return 10
    from sympy.stats.crv_types import NakagamiDistribution
    assert _test_args(NakagamiDistribution(1, 1))

def test_sympy__stats__crv_types__NormalDistribution():
    if False:
        for i in range(10):
            print('nop')
    from sympy.stats.crv_types import NormalDistribution
    assert _test_args(NormalDistribution(0, 1))

def test_sympy__stats__crv_types__GaussianInverseDistribution():
    if False:
        i = 10
        return i + 15
    from sympy.stats.crv_types import GaussianInverseDistribution
    assert _test_args(GaussianInverseDistribution(1, 1))

def test_sympy__stats__crv_types__ParetoDistribution():
    if False:
        i = 10
        return i + 15
    from sympy.stats.crv_types import ParetoDistribution
    assert _test_args(ParetoDistribution(1, 1))

def test_sympy__stats__crv_types__PowerFunctionDistribution():
    if False:
        return 10
    from sympy.stats.crv_types import PowerFunctionDistribution
    assert _test_args(PowerFunctionDistribution(2, 0, 1))

def test_sympy__stats__crv_types__QuadraticUDistribution():
    if False:
        while True:
            i = 10
    from sympy.stats.crv_types import QuadraticUDistribution
    assert _test_args(QuadraticUDistribution(1, 2))

def test_sympy__stats__crv_types__RaisedCosineDistribution():
    if False:
        return 10
    from sympy.stats.crv_types import RaisedCosineDistribution
    assert _test_args(RaisedCosineDistribution(1, 1))

def test_sympy__stats__crv_types__RayleighDistribution():
    if False:
        i = 10
        return i + 15
    from sympy.stats.crv_types import RayleighDistribution
    assert _test_args(RayleighDistribution(1))

def test_sympy__stats__crv_types__ReciprocalDistribution():
    if False:
        print('Hello World!')
    from sympy.stats.crv_types import ReciprocalDistribution
    assert _test_args(ReciprocalDistribution(5, 30))

def test_sympy__stats__crv_types__ShiftedGompertzDistribution():
    if False:
        i = 10
        return i + 15
    from sympy.stats.crv_types import ShiftedGompertzDistribution
    assert _test_args(ShiftedGompertzDistribution(1, 1))

def test_sympy__stats__crv_types__StudentTDistribution():
    if False:
        return 10
    from sympy.stats.crv_types import StudentTDistribution
    assert _test_args(StudentTDistribution(1))

def test_sympy__stats__crv_types__TrapezoidalDistribution():
    if False:
        for i in range(10):
            print('nop')
    from sympy.stats.crv_types import TrapezoidalDistribution
    assert _test_args(TrapezoidalDistribution(1, 2, 3, 4))

def test_sympy__stats__crv_types__TriangularDistribution():
    if False:
        return 10
    from sympy.stats.crv_types import TriangularDistribution
    assert _test_args(TriangularDistribution(-1, 0, 1))

def test_sympy__stats__crv_types__UniformDistribution():
    if False:
        while True:
            i = 10
    from sympy.stats.crv_types import UniformDistribution
    assert _test_args(UniformDistribution(0, 1))

def test_sympy__stats__crv_types__UniformSumDistribution():
    if False:
        for i in range(10):
            print('nop')
    from sympy.stats.crv_types import UniformSumDistribution
    assert _test_args(UniformSumDistribution(1))

def test_sympy__stats__crv_types__VonMisesDistribution():
    if False:
        print('Hello World!')
    from sympy.stats.crv_types import VonMisesDistribution
    assert _test_args(VonMisesDistribution(1, 1))

def test_sympy__stats__crv_types__WeibullDistribution():
    if False:
        for i in range(10):
            print('nop')
    from sympy.stats.crv_types import WeibullDistribution
    assert _test_args(WeibullDistribution(1, 1))

def test_sympy__stats__crv_types__WignerSemicircleDistribution():
    if False:
        print('Hello World!')
    from sympy.stats.crv_types import WignerSemicircleDistribution
    assert _test_args(WignerSemicircleDistribution(1))

def test_sympy__stats__drv_types__GeometricDistribution():
    if False:
        print('Hello World!')
    from sympy.stats.drv_types import GeometricDistribution
    assert _test_args(GeometricDistribution(0.5))

def test_sympy__stats__drv_types__HermiteDistribution():
    if False:
        print('Hello World!')
    from sympy.stats.drv_types import HermiteDistribution
    assert _test_args(HermiteDistribution(1, 2))

def test_sympy__stats__drv_types__LogarithmicDistribution():
    if False:
        while True:
            i = 10
    from sympy.stats.drv_types import LogarithmicDistribution
    assert _test_args(LogarithmicDistribution(0.5))

def test_sympy__stats__drv_types__NegativeBinomialDistribution():
    if False:
        i = 10
        return i + 15
    from sympy.stats.drv_types import NegativeBinomialDistribution
    assert _test_args(NegativeBinomialDistribution(0.5, 0.5))

def test_sympy__stats__drv_types__FlorySchulzDistribution():
    if False:
        i = 10
        return i + 15
    from sympy.stats.drv_types import FlorySchulzDistribution
    assert _test_args(FlorySchulzDistribution(0.5))

def test_sympy__stats__drv_types__PoissonDistribution():
    if False:
        for i in range(10):
            print('nop')
    from sympy.stats.drv_types import PoissonDistribution
    assert _test_args(PoissonDistribution(1))

def test_sympy__stats__drv_types__SkellamDistribution():
    if False:
        while True:
            i = 10
    from sympy.stats.drv_types import SkellamDistribution
    assert _test_args(SkellamDistribution(1, 1))

def test_sympy__stats__drv_types__YuleSimonDistribution():
    if False:
        for i in range(10):
            print('nop')
    from sympy.stats.drv_types import YuleSimonDistribution
    assert _test_args(YuleSimonDistribution(0.5))

def test_sympy__stats__drv_types__ZetaDistribution():
    if False:
        print('Hello World!')
    from sympy.stats.drv_types import ZetaDistribution
    assert _test_args(ZetaDistribution(1.5))

def test_sympy__stats__joint_rv__JointDistribution():
    if False:
        while True:
            i = 10
    from sympy.stats.joint_rv import JointDistribution
    assert _test_args(JointDistribution(1, 2, 3, 4))

def test_sympy__stats__joint_rv_types__MultivariateNormalDistribution():
    if False:
        print('Hello World!')
    from sympy.stats.joint_rv_types import MultivariateNormalDistribution
    assert _test_args(MultivariateNormalDistribution([0, 1], [[1, 0], [0, 1]]))

def test_sympy__stats__joint_rv_types__MultivariateLaplaceDistribution():
    if False:
        print('Hello World!')
    from sympy.stats.joint_rv_types import MultivariateLaplaceDistribution
    assert _test_args(MultivariateLaplaceDistribution([0, 1], [[1, 0], [0, 1]]))

def test_sympy__stats__joint_rv_types__MultivariateTDistribution():
    if False:
        i = 10
        return i + 15
    from sympy.stats.joint_rv_types import MultivariateTDistribution
    assert _test_args(MultivariateTDistribution([0, 1], [[1, 0], [0, 1]], 1))

def test_sympy__stats__joint_rv_types__NormalGammaDistribution():
    if False:
        while True:
            i = 10
    from sympy.stats.joint_rv_types import NormalGammaDistribution
    assert _test_args(NormalGammaDistribution(1, 2, 3, 4))

def test_sympy__stats__joint_rv_types__GeneralizedMultivariateLogGammaDistribution():
    if False:
        while True:
            i = 10
    from sympy.stats.joint_rv_types import GeneralizedMultivariateLogGammaDistribution
    (v, l, mu) = (4, [1, 2, 3, 4], [1, 2, 3, 4])
    assert _test_args(GeneralizedMultivariateLogGammaDistribution(S.Half, v, l, mu))

def test_sympy__stats__joint_rv_types__MultivariateBetaDistribution():
    if False:
        for i in range(10):
            print('nop')
    from sympy.stats.joint_rv_types import MultivariateBetaDistribution
    assert _test_args(MultivariateBetaDistribution([1, 2, 3]))

def test_sympy__stats__joint_rv_types__MultivariateEwensDistribution():
    if False:
        print('Hello World!')
    from sympy.stats.joint_rv_types import MultivariateEwensDistribution
    assert _test_args(MultivariateEwensDistribution(5, 1))

def test_sympy__stats__joint_rv_types__MultinomialDistribution():
    if False:
        print('Hello World!')
    from sympy.stats.joint_rv_types import MultinomialDistribution
    assert _test_args(MultinomialDistribution(5, [0.5, 0.1, 0.3]))

def test_sympy__stats__joint_rv_types__NegativeMultinomialDistribution():
    if False:
        return 10
    from sympy.stats.joint_rv_types import NegativeMultinomialDistribution
    assert _test_args(NegativeMultinomialDistribution(5, [0.5, 0.1, 0.3]))

def test_sympy__stats__rv__RandomIndexedSymbol():
    if False:
        for i in range(10):
            print('nop')
    from sympy.stats.rv import RandomIndexedSymbol, pspace
    from sympy.stats.stochastic_process_types import DiscreteMarkovChain
    X = DiscreteMarkovChain('X')
    assert _test_args(RandomIndexedSymbol(X[0].symbol, pspace(X[0])))

def test_sympy__stats__rv__RandomMatrixSymbol():
    if False:
        while True:
            i = 10
    from sympy.stats.rv import RandomMatrixSymbol
    from sympy.stats.random_matrix import RandomMatrixPSpace
    pspace = RandomMatrixPSpace('P')
    assert _test_args(RandomMatrixSymbol('M', 3, 3, pspace))

def test_sympy__stats__stochastic_process__StochasticPSpace():
    if False:
        print('Hello World!')
    from sympy.stats.stochastic_process import StochasticPSpace
    from sympy.stats.stochastic_process_types import StochasticProcess
    from sympy.stats.frv_types import BernoulliDistribution
    assert _test_args(StochasticPSpace('Y', StochasticProcess('Y', [1, 2, 3]), BernoulliDistribution(S.Half, 1, 0)))

def test_sympy__stats__stochastic_process_types__StochasticProcess():
    if False:
        print('Hello World!')
    from sympy.stats.stochastic_process_types import StochasticProcess
    assert _test_args(StochasticProcess('Y', [1, 2, 3]))

def test_sympy__stats__stochastic_process_types__MarkovProcess():
    if False:
        for i in range(10):
            print('nop')
    from sympy.stats.stochastic_process_types import MarkovProcess
    assert _test_args(MarkovProcess('Y', [1, 2, 3]))

def test_sympy__stats__stochastic_process_types__DiscreteTimeStochasticProcess():
    if False:
        while True:
            i = 10
    from sympy.stats.stochastic_process_types import DiscreteTimeStochasticProcess
    assert _test_args(DiscreteTimeStochasticProcess('Y', [1, 2, 3]))

def test_sympy__stats__stochastic_process_types__ContinuousTimeStochasticProcess():
    if False:
        while True:
            i = 10
    from sympy.stats.stochastic_process_types import ContinuousTimeStochasticProcess
    assert _test_args(ContinuousTimeStochasticProcess('Y', [1, 2, 3]))

def test_sympy__stats__stochastic_process_types__TransitionMatrixOf():
    if False:
        return 10
    from sympy.stats.stochastic_process_types import TransitionMatrixOf, DiscreteMarkovChain
    from sympy.matrices.expressions.matexpr import MatrixSymbol
    DMC = DiscreteMarkovChain('Y')
    assert _test_args(TransitionMatrixOf(DMC, MatrixSymbol('T', 3, 3)))

def test_sympy__stats__stochastic_process_types__GeneratorMatrixOf():
    if False:
        i = 10
        return i + 15
    from sympy.stats.stochastic_process_types import GeneratorMatrixOf, ContinuousMarkovChain
    from sympy.matrices.expressions.matexpr import MatrixSymbol
    DMC = ContinuousMarkovChain('Y')
    assert _test_args(GeneratorMatrixOf(DMC, MatrixSymbol('T', 3, 3)))

def test_sympy__stats__stochastic_process_types__StochasticStateSpaceOf():
    if False:
        i = 10
        return i + 15
    from sympy.stats.stochastic_process_types import StochasticStateSpaceOf, DiscreteMarkovChain
    DMC = DiscreteMarkovChain('Y')
    assert _test_args(StochasticStateSpaceOf(DMC, [0, 1, 2]))

def test_sympy__stats__stochastic_process_types__DiscreteMarkovChain():
    if False:
        i = 10
        return i + 15
    from sympy.stats.stochastic_process_types import DiscreteMarkovChain
    from sympy.matrices.expressions.matexpr import MatrixSymbol
    assert _test_args(DiscreteMarkovChain('Y', [0, 1, 2], MatrixSymbol('T', 3, 3)))

def test_sympy__stats__stochastic_process_types__ContinuousMarkovChain():
    if False:
        return 10
    from sympy.stats.stochastic_process_types import ContinuousMarkovChain
    from sympy.matrices.expressions.matexpr import MatrixSymbol
    assert _test_args(ContinuousMarkovChain('Y', [0, 1, 2], MatrixSymbol('T', 3, 3)))

def test_sympy__stats__stochastic_process_types__BernoulliProcess():
    if False:
        print('Hello World!')
    from sympy.stats.stochastic_process_types import BernoulliProcess
    assert _test_args(BernoulliProcess('B', 0.5, 1, 0))

def test_sympy__stats__stochastic_process_types__CountingProcess():
    if False:
        i = 10
        return i + 15
    from sympy.stats.stochastic_process_types import CountingProcess
    assert _test_args(CountingProcess('C'))

def test_sympy__stats__stochastic_process_types__PoissonProcess():
    if False:
        while True:
            i = 10
    from sympy.stats.stochastic_process_types import PoissonProcess
    assert _test_args(PoissonProcess('X', 2))

def test_sympy__stats__stochastic_process_types__WienerProcess():
    if False:
        i = 10
        return i + 15
    from sympy.stats.stochastic_process_types import WienerProcess
    assert _test_args(WienerProcess('X'))

def test_sympy__stats__stochastic_process_types__GammaProcess():
    if False:
        while True:
            i = 10
    from sympy.stats.stochastic_process_types import GammaProcess
    assert _test_args(GammaProcess('X', 1, 2))

def test_sympy__stats__random_matrix__RandomMatrixPSpace():
    if False:
        while True:
            i = 10
    from sympy.stats.random_matrix import RandomMatrixPSpace
    from sympy.stats.random_matrix_models import RandomMatrixEnsembleModel
    model = RandomMatrixEnsembleModel('R', 3)
    assert _test_args(RandomMatrixPSpace('P', model=model))

def test_sympy__stats__random_matrix_models__RandomMatrixEnsembleModel():
    if False:
        for i in range(10):
            print('nop')
    from sympy.stats.random_matrix_models import RandomMatrixEnsembleModel
    assert _test_args(RandomMatrixEnsembleModel('R', 3))

def test_sympy__stats__random_matrix_models__GaussianEnsembleModel():
    if False:
        print('Hello World!')
    from sympy.stats.random_matrix_models import GaussianEnsembleModel
    assert _test_args(GaussianEnsembleModel('G', 3))

def test_sympy__stats__random_matrix_models__GaussianUnitaryEnsembleModel():
    if False:
        i = 10
        return i + 15
    from sympy.stats.random_matrix_models import GaussianUnitaryEnsembleModel
    assert _test_args(GaussianUnitaryEnsembleModel('U', 3))

def test_sympy__stats__random_matrix_models__GaussianOrthogonalEnsembleModel():
    if False:
        i = 10
        return i + 15
    from sympy.stats.random_matrix_models import GaussianOrthogonalEnsembleModel
    assert _test_args(GaussianOrthogonalEnsembleModel('U', 3))

def test_sympy__stats__random_matrix_models__GaussianSymplecticEnsembleModel():
    if False:
        print('Hello World!')
    from sympy.stats.random_matrix_models import GaussianSymplecticEnsembleModel
    assert _test_args(GaussianSymplecticEnsembleModel('U', 3))

def test_sympy__stats__random_matrix_models__CircularEnsembleModel():
    if False:
        print('Hello World!')
    from sympy.stats.random_matrix_models import CircularEnsembleModel
    assert _test_args(CircularEnsembleModel('C', 3))

def test_sympy__stats__random_matrix_models__CircularUnitaryEnsembleModel():
    if False:
        while True:
            i = 10
    from sympy.stats.random_matrix_models import CircularUnitaryEnsembleModel
    assert _test_args(CircularUnitaryEnsembleModel('U', 3))

def test_sympy__stats__random_matrix_models__CircularOrthogonalEnsembleModel():
    if False:
        i = 10
        return i + 15
    from sympy.stats.random_matrix_models import CircularOrthogonalEnsembleModel
    assert _test_args(CircularOrthogonalEnsembleModel('O', 3))

def test_sympy__stats__random_matrix_models__CircularSymplecticEnsembleModel():
    if False:
        return 10
    from sympy.stats.random_matrix_models import CircularSymplecticEnsembleModel
    assert _test_args(CircularSymplecticEnsembleModel('S', 3))

def test_sympy__stats__symbolic_multivariate_probability__ExpectationMatrix():
    if False:
        i = 10
        return i + 15
    from sympy.stats import ExpectationMatrix
    from sympy.stats.rv import RandomMatrixSymbol
    assert _test_args(ExpectationMatrix(RandomMatrixSymbol('R', 2, 1)))

def test_sympy__stats__symbolic_multivariate_probability__VarianceMatrix():
    if False:
        while True:
            i = 10
    from sympy.stats import VarianceMatrix
    from sympy.stats.rv import RandomMatrixSymbol
    assert _test_args(VarianceMatrix(RandomMatrixSymbol('R', 3, 1)))

def test_sympy__stats__symbolic_multivariate_probability__CrossCovarianceMatrix():
    if False:
        print('Hello World!')
    from sympy.stats import CrossCovarianceMatrix
    from sympy.stats.rv import RandomMatrixSymbol
    assert _test_args(CrossCovarianceMatrix(RandomMatrixSymbol('R', 3, 1), RandomMatrixSymbol('X', 3, 1)))

def test_sympy__stats__matrix_distributions__MatrixPSpace():
    if False:
        print('Hello World!')
    from sympy.stats.matrix_distributions import MatrixDistribution, MatrixPSpace
    from sympy.matrices.dense import Matrix
    M = MatrixDistribution(1, Matrix([[1, 0], [0, 1]]))
    assert _test_args(MatrixPSpace('M', M, 2, 2))

def test_sympy__stats__matrix_distributions__MatrixDistribution():
    if False:
        while True:
            i = 10
    from sympy.stats.matrix_distributions import MatrixDistribution
    from sympy.matrices.dense import Matrix
    assert _test_args(MatrixDistribution(1, Matrix([[1, 0], [0, 1]])))

def test_sympy__stats__matrix_distributions__MatrixGammaDistribution():
    if False:
        print('Hello World!')
    from sympy.stats.matrix_distributions import MatrixGammaDistribution
    from sympy.matrices.dense import Matrix
    assert _test_args(MatrixGammaDistribution(3, 4, Matrix([[1, 0], [0, 1]])))

def test_sympy__stats__matrix_distributions__WishartDistribution():
    if False:
        i = 10
        return i + 15
    from sympy.stats.matrix_distributions import WishartDistribution
    from sympy.matrices.dense import Matrix
    assert _test_args(WishartDistribution(3, Matrix([[1, 0], [0, 1]])))

def test_sympy__stats__matrix_distributions__MatrixNormalDistribution():
    if False:
        print('Hello World!')
    from sympy.stats.matrix_distributions import MatrixNormalDistribution
    from sympy.matrices.expressions.matexpr import MatrixSymbol
    L = MatrixSymbol('L', 1, 2)
    S1 = MatrixSymbol('S1', 1, 1)
    S2 = MatrixSymbol('S2', 2, 2)
    assert _test_args(MatrixNormalDistribution(L, S1, S2))

def test_sympy__stats__matrix_distributions__MatrixStudentTDistribution():
    if False:
        while True:
            i = 10
    from sympy.stats.matrix_distributions import MatrixStudentTDistribution
    from sympy.matrices.expressions.matexpr import MatrixSymbol
    v = symbols('v', positive=True)
    Omega = MatrixSymbol('Omega', 3, 3)
    Sigma = MatrixSymbol('Sigma', 1, 1)
    Location = MatrixSymbol('Location', 1, 3)
    assert _test_args(MatrixStudentTDistribution(v, Location, Omega, Sigma))

def test_sympy__utilities__matchpy_connector__WildDot():
    if False:
        return 10
    from sympy.utilities.matchpy_connector import WildDot
    assert _test_args(WildDot('w_'))

def test_sympy__utilities__matchpy_connector__WildPlus():
    if False:
        return 10
    from sympy.utilities.matchpy_connector import WildPlus
    assert _test_args(WildPlus('w__'))

def test_sympy__utilities__matchpy_connector__WildStar():
    if False:
        i = 10
        return i + 15
    from sympy.utilities.matchpy_connector import WildStar
    assert _test_args(WildStar('w___'))

def test_sympy__core__symbol__Str():
    if False:
        return 10
    from sympy.core.symbol import Str
    assert _test_args(Str('t'))

def test_sympy__core__symbol__Dummy():
    if False:
        return 10
    from sympy.core.symbol import Dummy
    assert _test_args(Dummy('t'))

def test_sympy__core__symbol__Symbol():
    if False:
        return 10
    from sympy.core.symbol import Symbol
    assert _test_args(Symbol('t'))

def test_sympy__core__symbol__Wild():
    if False:
        while True:
            i = 10
    from sympy.core.symbol import Wild
    assert _test_args(Wild('x', exclude=[x]))

@SKIP('abstract class')
def test_sympy__functions__combinatorial__factorials__CombinatorialFunction():
    if False:
        for i in range(10):
            print('nop')
    pass

def test_sympy__functions__combinatorial__factorials__FallingFactorial():
    if False:
        return 10
    from sympy.functions.combinatorial.factorials import FallingFactorial
    assert _test_args(FallingFactorial(2, x))

def test_sympy__functions__combinatorial__factorials__MultiFactorial():
    if False:
        i = 10
        return i + 15
    from sympy.functions.combinatorial.factorials import MultiFactorial
    assert _test_args(MultiFactorial(x))

def test_sympy__functions__combinatorial__factorials__RisingFactorial():
    if False:
        for i in range(10):
            print('nop')
    from sympy.functions.combinatorial.factorials import RisingFactorial
    assert _test_args(RisingFactorial(2, x))

def test_sympy__functions__combinatorial__factorials__binomial():
    if False:
        return 10
    from sympy.functions.combinatorial.factorials import binomial
    assert _test_args(binomial(2, x))

def test_sympy__functions__combinatorial__factorials__subfactorial():
    if False:
        print('Hello World!')
    from sympy.functions.combinatorial.factorials import subfactorial
    assert _test_args(subfactorial(x))

def test_sympy__functions__combinatorial__factorials__factorial():
    if False:
        print('Hello World!')
    from sympy.functions.combinatorial.factorials import factorial
    assert _test_args(factorial(x))

def test_sympy__functions__combinatorial__factorials__factorial2():
    if False:
        print('Hello World!')
    from sympy.functions.combinatorial.factorials import factorial2
    assert _test_args(factorial2(x))

def test_sympy__functions__combinatorial__numbers__bell():
    if False:
        return 10
    from sympy.functions.combinatorial.numbers import bell
    assert _test_args(bell(x, y))

def test_sympy__functions__combinatorial__numbers__bernoulli():
    if False:
        return 10
    from sympy.functions.combinatorial.numbers import bernoulli
    assert _test_args(bernoulli(x))

def test_sympy__functions__combinatorial__numbers__catalan():
    if False:
        for i in range(10):
            print('nop')
    from sympy.functions.combinatorial.numbers import catalan
    assert _test_args(catalan(x))

def test_sympy__functions__combinatorial__numbers__genocchi():
    if False:
        return 10
    from sympy.functions.combinatorial.numbers import genocchi
    assert _test_args(genocchi(x))

def test_sympy__functions__combinatorial__numbers__euler():
    if False:
        return 10
    from sympy.functions.combinatorial.numbers import euler
    assert _test_args(euler(x))

def test_sympy__functions__combinatorial__numbers__andre():
    if False:
        print('Hello World!')
    from sympy.functions.combinatorial.numbers import andre
    assert _test_args(andre(x))

def test_sympy__functions__combinatorial__numbers__carmichael():
    if False:
        print('Hello World!')
    from sympy.functions.combinatorial.numbers import carmichael
    assert _test_args(carmichael(x))

def test_sympy__functions__combinatorial__numbers__motzkin():
    if False:
        while True:
            i = 10
    from sympy.functions.combinatorial.numbers import motzkin
    assert _test_args(motzkin(5))

def test_sympy__functions__combinatorial__numbers__fibonacci():
    if False:
        i = 10
        return i + 15
    from sympy.functions.combinatorial.numbers import fibonacci
    assert _test_args(fibonacci(x))

def test_sympy__functions__combinatorial__numbers__tribonacci():
    if False:
        return 10
    from sympy.functions.combinatorial.numbers import tribonacci
    assert _test_args(tribonacci(x))

def test_sympy__functions__combinatorial__numbers__harmonic():
    if False:
        print('Hello World!')
    from sympy.functions.combinatorial.numbers import harmonic
    assert _test_args(harmonic(x, 2))

def test_sympy__functions__combinatorial__numbers__lucas():
    if False:
        print('Hello World!')
    from sympy.functions.combinatorial.numbers import lucas
    assert _test_args(lucas(x))

def test_sympy__functions__combinatorial__numbers__partition():
    if False:
        i = 10
        return i + 15
    from sympy.core.symbol import Symbol
    from sympy.functions.combinatorial.numbers import partition
    assert _test_args(partition(Symbol('a', integer=True)))

def test_sympy__functions__elementary__complexes__Abs():
    if False:
        print('Hello World!')
    from sympy.functions.elementary.complexes import Abs
    assert _test_args(Abs(x))

def test_sympy__functions__elementary__complexes__adjoint():
    if False:
        while True:
            i = 10
    from sympy.functions.elementary.complexes import adjoint
    assert _test_args(adjoint(x))

def test_sympy__functions__elementary__complexes__arg():
    if False:
        i = 10
        return i + 15
    from sympy.functions.elementary.complexes import arg
    assert _test_args(arg(x))

def test_sympy__functions__elementary__complexes__conjugate():
    if False:
        return 10
    from sympy.functions.elementary.complexes import conjugate
    assert _test_args(conjugate(x))

def test_sympy__functions__elementary__complexes__im():
    if False:
        print('Hello World!')
    from sympy.functions.elementary.complexes import im
    assert _test_args(im(x))

def test_sympy__functions__elementary__complexes__re():
    if False:
        return 10
    from sympy.functions.elementary.complexes import re
    assert _test_args(re(x))

def test_sympy__functions__elementary__complexes__sign():
    if False:
        for i in range(10):
            print('nop')
    from sympy.functions.elementary.complexes import sign
    assert _test_args(sign(x))

def test_sympy__functions__elementary__complexes__polar_lift():
    if False:
        for i in range(10):
            print('nop')
    from sympy.functions.elementary.complexes import polar_lift
    assert _test_args(polar_lift(x))

def test_sympy__functions__elementary__complexes__periodic_argument():
    if False:
        return 10
    from sympy.functions.elementary.complexes import periodic_argument
    assert _test_args(periodic_argument(x, y))

def test_sympy__functions__elementary__complexes__principal_branch():
    if False:
        print('Hello World!')
    from sympy.functions.elementary.complexes import principal_branch
    assert _test_args(principal_branch(x, y))

def test_sympy__functions__elementary__complexes__transpose():
    if False:
        for i in range(10):
            print('nop')
    from sympy.functions.elementary.complexes import transpose
    assert _test_args(transpose(x))

def test_sympy__functions__elementary__exponential__LambertW():
    if False:
        while True:
            i = 10
    from sympy.functions.elementary.exponential import LambertW
    assert _test_args(LambertW(2))

@SKIP('abstract class')
def test_sympy__functions__elementary__exponential__ExpBase():
    if False:
        for i in range(10):
            print('nop')
    pass

def test_sympy__functions__elementary__exponential__exp():
    if False:
        return 10
    from sympy.functions.elementary.exponential import exp
    assert _test_args(exp(2))

def test_sympy__functions__elementary__exponential__exp_polar():
    if False:
        return 10
    from sympy.functions.elementary.exponential import exp_polar
    assert _test_args(exp_polar(2))

def test_sympy__functions__elementary__exponential__log():
    if False:
        print('Hello World!')
    from sympy.functions.elementary.exponential import log
    assert _test_args(log(2))

@SKIP('abstract class')
def test_sympy__functions__elementary__hyperbolic__HyperbolicFunction():
    if False:
        print('Hello World!')
    pass

@SKIP('abstract class')
def test_sympy__functions__elementary__hyperbolic__ReciprocalHyperbolicFunction():
    if False:
        return 10
    pass

@SKIP('abstract class')
def test_sympy__functions__elementary__hyperbolic__InverseHyperbolicFunction():
    if False:
        while True:
            i = 10
    pass

def test_sympy__functions__elementary__hyperbolic__acosh():
    if False:
        i = 10
        return i + 15
    from sympy.functions.elementary.hyperbolic import acosh
    assert _test_args(acosh(2))

def test_sympy__functions__elementary__hyperbolic__acoth():
    if False:
        while True:
            i = 10
    from sympy.functions.elementary.hyperbolic import acoth
    assert _test_args(acoth(2))

def test_sympy__functions__elementary__hyperbolic__asinh():
    if False:
        return 10
    from sympy.functions.elementary.hyperbolic import asinh
    assert _test_args(asinh(2))

def test_sympy__functions__elementary__hyperbolic__atanh():
    if False:
        print('Hello World!')
    from sympy.functions.elementary.hyperbolic import atanh
    assert _test_args(atanh(2))

def test_sympy__functions__elementary__hyperbolic__asech():
    if False:
        for i in range(10):
            print('nop')
    from sympy.functions.elementary.hyperbolic import asech
    assert _test_args(asech(x))

def test_sympy__functions__elementary__hyperbolic__acsch():
    if False:
        print('Hello World!')
    from sympy.functions.elementary.hyperbolic import acsch
    assert _test_args(acsch(x))

def test_sympy__functions__elementary__hyperbolic__cosh():
    if False:
        print('Hello World!')
    from sympy.functions.elementary.hyperbolic import cosh
    assert _test_args(cosh(2))

def test_sympy__functions__elementary__hyperbolic__coth():
    if False:
        return 10
    from sympy.functions.elementary.hyperbolic import coth
    assert _test_args(coth(2))

def test_sympy__functions__elementary__hyperbolic__csch():
    if False:
        print('Hello World!')
    from sympy.functions.elementary.hyperbolic import csch
    assert _test_args(csch(2))

def test_sympy__functions__elementary__hyperbolic__sech():
    if False:
        for i in range(10):
            print('nop')
    from sympy.functions.elementary.hyperbolic import sech
    assert _test_args(sech(2))

def test_sympy__functions__elementary__hyperbolic__sinh():
    if False:
        print('Hello World!')
    from sympy.functions.elementary.hyperbolic import sinh
    assert _test_args(sinh(2))

def test_sympy__functions__elementary__hyperbolic__tanh():
    if False:
        i = 10
        return i + 15
    from sympy.functions.elementary.hyperbolic import tanh
    assert _test_args(tanh(2))

@SKIP('abstract class')
def test_sympy__functions__elementary__integers__RoundFunction():
    if False:
        return 10
    pass

def test_sympy__functions__elementary__integers__ceiling():
    if False:
        i = 10
        return i + 15
    from sympy.functions.elementary.integers import ceiling
    assert _test_args(ceiling(x))

def test_sympy__functions__elementary__integers__floor():
    if False:
        print('Hello World!')
    from sympy.functions.elementary.integers import floor
    assert _test_args(floor(x))

def test_sympy__functions__elementary__integers__frac():
    if False:
        while True:
            i = 10
    from sympy.functions.elementary.integers import frac
    assert _test_args(frac(x))

def test_sympy__functions__elementary__miscellaneous__IdentityFunction():
    if False:
        print('Hello World!')
    from sympy.functions.elementary.miscellaneous import IdentityFunction
    assert _test_args(IdentityFunction())

def test_sympy__functions__elementary__miscellaneous__Max():
    if False:
        for i in range(10):
            print('nop')
    from sympy.functions.elementary.miscellaneous import Max
    assert _test_args(Max(x, 2))

def test_sympy__functions__elementary__miscellaneous__Min():
    if False:
        for i in range(10):
            print('nop')
    from sympy.functions.elementary.miscellaneous import Min
    assert _test_args(Min(x, 2))

@SKIP('abstract class')
def test_sympy__functions__elementary__miscellaneous__MinMaxBase():
    if False:
        while True:
            i = 10
    pass

def test_sympy__functions__elementary__miscellaneous__Rem():
    if False:
        print('Hello World!')
    from sympy.functions.elementary.miscellaneous import Rem
    assert _test_args(Rem(x, 2))

def test_sympy__functions__elementary__piecewise__ExprCondPair():
    if False:
        for i in range(10):
            print('nop')
    from sympy.functions.elementary.piecewise import ExprCondPair
    assert _test_args(ExprCondPair(1, True))

def test_sympy__functions__elementary__piecewise__Piecewise():
    if False:
        return 10
    from sympy.functions.elementary.piecewise import Piecewise
    assert _test_args(Piecewise((1, x >= 0), (0, True)))

@SKIP('abstract class')
def test_sympy__functions__elementary__trigonometric__TrigonometricFunction():
    if False:
        return 10
    pass

@SKIP('abstract class')
def test_sympy__functions__elementary__trigonometric__ReciprocalTrigonometricFunction():
    if False:
        print('Hello World!')
    pass

@SKIP('abstract class')
def test_sympy__functions__elementary__trigonometric__InverseTrigonometricFunction():
    if False:
        for i in range(10):
            print('nop')
    pass

def test_sympy__functions__elementary__trigonometric__acos():
    if False:
        return 10
    from sympy.functions.elementary.trigonometric import acos
    assert _test_args(acos(2))

def test_sympy__functions__elementary__trigonometric__acot():
    if False:
        while True:
            i = 10
    from sympy.functions.elementary.trigonometric import acot
    assert _test_args(acot(2))

def test_sympy__functions__elementary__trigonometric__asin():
    if False:
        return 10
    from sympy.functions.elementary.trigonometric import asin
    assert _test_args(asin(2))

def test_sympy__functions__elementary__trigonometric__asec():
    if False:
        print('Hello World!')
    from sympy.functions.elementary.trigonometric import asec
    assert _test_args(asec(x))

def test_sympy__functions__elementary__trigonometric__acsc():
    if False:
        print('Hello World!')
    from sympy.functions.elementary.trigonometric import acsc
    assert _test_args(acsc(x))

def test_sympy__functions__elementary__trigonometric__atan():
    if False:
        return 10
    from sympy.functions.elementary.trigonometric import atan
    assert _test_args(atan(2))

def test_sympy__functions__elementary__trigonometric__atan2():
    if False:
        while True:
            i = 10
    from sympy.functions.elementary.trigonometric import atan2
    assert _test_args(atan2(2, 3))

def test_sympy__functions__elementary__trigonometric__cos():
    if False:
        i = 10
        return i + 15
    from sympy.functions.elementary.trigonometric import cos
    assert _test_args(cos(2))

def test_sympy__functions__elementary__trigonometric__csc():
    if False:
        while True:
            i = 10
    from sympy.functions.elementary.trigonometric import csc
    assert _test_args(csc(2))

def test_sympy__functions__elementary__trigonometric__cot():
    if False:
        for i in range(10):
            print('nop')
    from sympy.functions.elementary.trigonometric import cot
    assert _test_args(cot(2))

def test_sympy__functions__elementary__trigonometric__sin():
    if False:
        return 10
    assert _test_args(sin(2))

def test_sympy__functions__elementary__trigonometric__sinc():
    if False:
        return 10
    from sympy.functions.elementary.trigonometric import sinc
    assert _test_args(sinc(2))

def test_sympy__functions__elementary__trigonometric__sec():
    if False:
        return 10
    from sympy.functions.elementary.trigonometric import sec
    assert _test_args(sec(2))

def test_sympy__functions__elementary__trigonometric__tan():
    if False:
        while True:
            i = 10
    from sympy.functions.elementary.trigonometric import tan
    assert _test_args(tan(2))

@SKIP('abstract class')
def test_sympy__functions__special__bessel__BesselBase():
    if False:
        for i in range(10):
            print('nop')
    pass

@SKIP('abstract class')
def test_sympy__functions__special__bessel__SphericalBesselBase():
    if False:
        print('Hello World!')
    pass

@SKIP('abstract class')
def test_sympy__functions__special__bessel__SphericalHankelBase():
    if False:
        while True:
            i = 10
    pass

def test_sympy__functions__special__bessel__besseli():
    if False:
        return 10
    from sympy.functions.special.bessel import besseli
    assert _test_args(besseli(x, 1))

def test_sympy__functions__special__bessel__besselj():
    if False:
        while True:
            i = 10
    from sympy.functions.special.bessel import besselj
    assert _test_args(besselj(x, 1))

def test_sympy__functions__special__bessel__besselk():
    if False:
        print('Hello World!')
    from sympy.functions.special.bessel import besselk
    assert _test_args(besselk(x, 1))

def test_sympy__functions__special__bessel__bessely():
    if False:
        while True:
            i = 10
    from sympy.functions.special.bessel import bessely
    assert _test_args(bessely(x, 1))

def test_sympy__functions__special__bessel__hankel1():
    if False:
        for i in range(10):
            print('nop')
    from sympy.functions.special.bessel import hankel1
    assert _test_args(hankel1(x, 1))

def test_sympy__functions__special__bessel__hankel2():
    if False:
        print('Hello World!')
    from sympy.functions.special.bessel import hankel2
    assert _test_args(hankel2(x, 1))

def test_sympy__functions__special__bessel__jn():
    if False:
        return 10
    from sympy.functions.special.bessel import jn
    assert _test_args(jn(0, x))

def test_sympy__functions__special__bessel__yn():
    if False:
        for i in range(10):
            print('nop')
    from sympy.functions.special.bessel import yn
    assert _test_args(yn(0, x))

def test_sympy__functions__special__bessel__hn1():
    if False:
        while True:
            i = 10
    from sympy.functions.special.bessel import hn1
    assert _test_args(hn1(0, x))

def test_sympy__functions__special__bessel__hn2():
    if False:
        return 10
    from sympy.functions.special.bessel import hn2
    assert _test_args(hn2(0, x))

def test_sympy__functions__special__bessel__AiryBase():
    if False:
        return 10
    pass

def test_sympy__functions__special__bessel__airyai():
    if False:
        i = 10
        return i + 15
    from sympy.functions.special.bessel import airyai
    assert _test_args(airyai(2))

def test_sympy__functions__special__bessel__airybi():
    if False:
        while True:
            i = 10
    from sympy.functions.special.bessel import airybi
    assert _test_args(airybi(2))

def test_sympy__functions__special__bessel__airyaiprime():
    if False:
        return 10
    from sympy.functions.special.bessel import airyaiprime
    assert _test_args(airyaiprime(2))

def test_sympy__functions__special__bessel__airybiprime():
    if False:
        print('Hello World!')
    from sympy.functions.special.bessel import airybiprime
    assert _test_args(airybiprime(2))

def test_sympy__functions__special__bessel__marcumq():
    if False:
        print('Hello World!')
    from sympy.functions.special.bessel import marcumq
    assert _test_args(marcumq(x, y, z))

def test_sympy__functions__special__elliptic_integrals__elliptic_k():
    if False:
        return 10
    from sympy.functions.special.elliptic_integrals import elliptic_k as K
    assert _test_args(K(x))

def test_sympy__functions__special__elliptic_integrals__elliptic_f():
    if False:
        i = 10
        return i + 15
    from sympy.functions.special.elliptic_integrals import elliptic_f as F
    assert _test_args(F(x, y))

def test_sympy__functions__special__elliptic_integrals__elliptic_e():
    if False:
        return 10
    from sympy.functions.special.elliptic_integrals import elliptic_e as E
    assert _test_args(E(x))
    assert _test_args(E(x, y))

def test_sympy__functions__special__elliptic_integrals__elliptic_pi():
    if False:
        while True:
            i = 10
    from sympy.functions.special.elliptic_integrals import elliptic_pi as P
    assert _test_args(P(x, y))
    assert _test_args(P(x, y, z))

def test_sympy__functions__special__delta_functions__DiracDelta():
    if False:
        i = 10
        return i + 15
    from sympy.functions.special.delta_functions import DiracDelta
    assert _test_args(DiracDelta(x, 1))

def test_sympy__functions__special__singularity_functions__SingularityFunction():
    if False:
        print('Hello World!')
    from sympy.functions.special.singularity_functions import SingularityFunction
    assert _test_args(SingularityFunction(x, y, z))

def test_sympy__functions__special__delta_functions__Heaviside():
    if False:
        return 10
    from sympy.functions.special.delta_functions import Heaviside
    assert _test_args(Heaviside(x))

def test_sympy__functions__special__error_functions__erf():
    if False:
        print('Hello World!')
    from sympy.functions.special.error_functions import erf
    assert _test_args(erf(2))

def test_sympy__functions__special__error_functions__erfc():
    if False:
        print('Hello World!')
    from sympy.functions.special.error_functions import erfc
    assert _test_args(erfc(2))

def test_sympy__functions__special__error_functions__erfi():
    if False:
        for i in range(10):
            print('nop')
    from sympy.functions.special.error_functions import erfi
    assert _test_args(erfi(2))

def test_sympy__functions__special__error_functions__erf2():
    if False:
        print('Hello World!')
    from sympy.functions.special.error_functions import erf2
    assert _test_args(erf2(2, 3))

def test_sympy__functions__special__error_functions__erfinv():
    if False:
        while True:
            i = 10
    from sympy.functions.special.error_functions import erfinv
    assert _test_args(erfinv(2))

def test_sympy__functions__special__error_functions__erfcinv():
    if False:
        i = 10
        return i + 15
    from sympy.functions.special.error_functions import erfcinv
    assert _test_args(erfcinv(2))

def test_sympy__functions__special__error_functions__erf2inv():
    if False:
        print('Hello World!')
    from sympy.functions.special.error_functions import erf2inv
    assert _test_args(erf2inv(2, 3))

@SKIP('abstract class')
def test_sympy__functions__special__error_functions__FresnelIntegral():
    if False:
        i = 10
        return i + 15
    pass

def test_sympy__functions__special__error_functions__fresnels():
    if False:
        for i in range(10):
            print('nop')
    from sympy.functions.special.error_functions import fresnels
    assert _test_args(fresnels(2))

def test_sympy__functions__special__error_functions__fresnelc():
    if False:
        i = 10
        return i + 15
    from sympy.functions.special.error_functions import fresnelc
    assert _test_args(fresnelc(2))

def test_sympy__functions__special__error_functions__erfs():
    if False:
        while True:
            i = 10
    from sympy.functions.special.error_functions import _erfs
    assert _test_args(_erfs(2))

def test_sympy__functions__special__error_functions__Ei():
    if False:
        while True:
            i = 10
    from sympy.functions.special.error_functions import Ei
    assert _test_args(Ei(2))

def test_sympy__functions__special__error_functions__li():
    if False:
        print('Hello World!')
    from sympy.functions.special.error_functions import li
    assert _test_args(li(2))

def test_sympy__functions__special__error_functions__Li():
    if False:
        for i in range(10):
            print('nop')
    from sympy.functions.special.error_functions import Li
    assert _test_args(Li(5))

@SKIP('abstract class')
def test_sympy__functions__special__error_functions__TrigonometricIntegral():
    if False:
        while True:
            i = 10
    pass

def test_sympy__functions__special__error_functions__Si():
    if False:
        print('Hello World!')
    from sympy.functions.special.error_functions import Si
    assert _test_args(Si(2))

def test_sympy__functions__special__error_functions__Ci():
    if False:
        return 10
    from sympy.functions.special.error_functions import Ci
    assert _test_args(Ci(2))

def test_sympy__functions__special__error_functions__Shi():
    if False:
        print('Hello World!')
    from sympy.functions.special.error_functions import Shi
    assert _test_args(Shi(2))

def test_sympy__functions__special__error_functions__Chi():
    if False:
        for i in range(10):
            print('nop')
    from sympy.functions.special.error_functions import Chi
    assert _test_args(Chi(2))

def test_sympy__functions__special__error_functions__expint():
    if False:
        i = 10
        return i + 15
    from sympy.functions.special.error_functions import expint
    assert _test_args(expint(y, x))

def test_sympy__functions__special__gamma_functions__gamma():
    if False:
        while True:
            i = 10
    from sympy.functions.special.gamma_functions import gamma
    assert _test_args(gamma(x))

def test_sympy__functions__special__gamma_functions__loggamma():
    if False:
        print('Hello World!')
    from sympy.functions.special.gamma_functions import loggamma
    assert _test_args(loggamma(x))

def test_sympy__functions__special__gamma_functions__lowergamma():
    if False:
        for i in range(10):
            print('nop')
    from sympy.functions.special.gamma_functions import lowergamma
    assert _test_args(lowergamma(x, 2))

def test_sympy__functions__special__gamma_functions__polygamma():
    if False:
        return 10
    from sympy.functions.special.gamma_functions import polygamma
    assert _test_args(polygamma(x, 2))

def test_sympy__functions__special__gamma_functions__digamma():
    if False:
        return 10
    from sympy.functions.special.gamma_functions import digamma
    assert _test_args(digamma(x))

def test_sympy__functions__special__gamma_functions__trigamma():
    if False:
        return 10
    from sympy.functions.special.gamma_functions import trigamma
    assert _test_args(trigamma(x))

def test_sympy__functions__special__gamma_functions__uppergamma():
    if False:
        print('Hello World!')
    from sympy.functions.special.gamma_functions import uppergamma
    assert _test_args(uppergamma(x, 2))

def test_sympy__functions__special__gamma_functions__multigamma():
    if False:
        i = 10
        return i + 15
    from sympy.functions.special.gamma_functions import multigamma
    assert _test_args(multigamma(x, 1))

def test_sympy__functions__special__beta_functions__beta():
    if False:
        while True:
            i = 10
    from sympy.functions.special.beta_functions import beta
    assert _test_args(beta(x))
    assert _test_args(beta(x, x))

def test_sympy__functions__special__beta_functions__betainc():
    if False:
        return 10
    from sympy.functions.special.beta_functions import betainc
    assert _test_args(betainc(a, b, x, y))

def test_sympy__functions__special__beta_functions__betainc_regularized():
    if False:
        print('Hello World!')
    from sympy.functions.special.beta_functions import betainc_regularized
    assert _test_args(betainc_regularized(a, b, x, y))

def test_sympy__functions__special__mathieu_functions__MathieuBase():
    if False:
        print('Hello World!')
    pass

def test_sympy__functions__special__mathieu_functions__mathieus():
    if False:
        print('Hello World!')
    from sympy.functions.special.mathieu_functions import mathieus
    assert _test_args(mathieus(1, 1, 1))

def test_sympy__functions__special__mathieu_functions__mathieuc():
    if False:
        return 10
    from sympy.functions.special.mathieu_functions import mathieuc
    assert _test_args(mathieuc(1, 1, 1))

def test_sympy__functions__special__mathieu_functions__mathieusprime():
    if False:
        i = 10
        return i + 15
    from sympy.functions.special.mathieu_functions import mathieusprime
    assert _test_args(mathieusprime(1, 1, 1))

def test_sympy__functions__special__mathieu_functions__mathieucprime():
    if False:
        print('Hello World!')
    from sympy.functions.special.mathieu_functions import mathieucprime
    assert _test_args(mathieucprime(1, 1, 1))

@SKIP('abstract class')
def test_sympy__functions__special__hyper__TupleParametersBase():
    if False:
        print('Hello World!')
    pass

@SKIP('abstract class')
def test_sympy__functions__special__hyper__TupleArg():
    if False:
        i = 10
        return i + 15
    pass

def test_sympy__functions__special__hyper__hyper():
    if False:
        i = 10
        return i + 15
    from sympy.functions.special.hyper import hyper
    assert _test_args(hyper([1, 2, 3], [4, 5], x))

def test_sympy__functions__special__hyper__meijerg():
    if False:
        for i in range(10):
            print('nop')
    from sympy.functions.special.hyper import meijerg
    assert _test_args(meijerg([1, 2, 3], [4, 5], [6], [], x))

@SKIP('abstract class')
def test_sympy__functions__special__hyper__HyperRep():
    if False:
        i = 10
        return i + 15
    pass

def test_sympy__functions__special__hyper__HyperRep_power1():
    if False:
        for i in range(10):
            print('nop')
    from sympy.functions.special.hyper import HyperRep_power1
    assert _test_args(HyperRep_power1(x, y))

def test_sympy__functions__special__hyper__HyperRep_power2():
    if False:
        for i in range(10):
            print('nop')
    from sympy.functions.special.hyper import HyperRep_power2
    assert _test_args(HyperRep_power2(x, y))

def test_sympy__functions__special__hyper__HyperRep_log1():
    if False:
        print('Hello World!')
    from sympy.functions.special.hyper import HyperRep_log1
    assert _test_args(HyperRep_log1(x))

def test_sympy__functions__special__hyper__HyperRep_atanh():
    if False:
        return 10
    from sympy.functions.special.hyper import HyperRep_atanh
    assert _test_args(HyperRep_atanh(x))

def test_sympy__functions__special__hyper__HyperRep_asin1():
    if False:
        print('Hello World!')
    from sympy.functions.special.hyper import HyperRep_asin1
    assert _test_args(HyperRep_asin1(x))

def test_sympy__functions__special__hyper__HyperRep_asin2():
    if False:
        i = 10
        return i + 15
    from sympy.functions.special.hyper import HyperRep_asin2
    assert _test_args(HyperRep_asin2(x))

def test_sympy__functions__special__hyper__HyperRep_sqrts1():
    if False:
        while True:
            i = 10
    from sympy.functions.special.hyper import HyperRep_sqrts1
    assert _test_args(HyperRep_sqrts1(x, y))

def test_sympy__functions__special__hyper__HyperRep_sqrts2():
    if False:
        return 10
    from sympy.functions.special.hyper import HyperRep_sqrts2
    assert _test_args(HyperRep_sqrts2(x, y))

def test_sympy__functions__special__hyper__HyperRep_log2():
    if False:
        return 10
    from sympy.functions.special.hyper import HyperRep_log2
    assert _test_args(HyperRep_log2(x))

def test_sympy__functions__special__hyper__HyperRep_cosasin():
    if False:
        while True:
            i = 10
    from sympy.functions.special.hyper import HyperRep_cosasin
    assert _test_args(HyperRep_cosasin(x, y))

def test_sympy__functions__special__hyper__HyperRep_sinasin():
    if False:
        return 10
    from sympy.functions.special.hyper import HyperRep_sinasin
    assert _test_args(HyperRep_sinasin(x, y))

def test_sympy__functions__special__hyper__appellf1():
    if False:
        for i in range(10):
            print('nop')
    from sympy.functions.special.hyper import appellf1
    (a, b1, b2, c, x, y) = symbols('a b1 b2 c x y')
    assert _test_args(appellf1(a, b1, b2, c, x, y))

@SKIP('abstract class')
def test_sympy__functions__special__polynomials__OrthogonalPolynomial():
    if False:
        for i in range(10):
            print('nop')
    pass

def test_sympy__functions__special__polynomials__jacobi():
    if False:
        for i in range(10):
            print('nop')
    from sympy.functions.special.polynomials import jacobi
    assert _test_args(jacobi(x, y, 2, 2))

def test_sympy__functions__special__polynomials__gegenbauer():
    if False:
        print('Hello World!')
    from sympy.functions.special.polynomials import gegenbauer
    assert _test_args(gegenbauer(x, 2, 2))

def test_sympy__functions__special__polynomials__chebyshevt():
    if False:
        while True:
            i = 10
    from sympy.functions.special.polynomials import chebyshevt
    assert _test_args(chebyshevt(x, 2))

def test_sympy__functions__special__polynomials__chebyshevt_root():
    if False:
        for i in range(10):
            print('nop')
    from sympy.functions.special.polynomials import chebyshevt_root
    assert _test_args(chebyshevt_root(3, 2))

def test_sympy__functions__special__polynomials__chebyshevu():
    if False:
        return 10
    from sympy.functions.special.polynomials import chebyshevu
    assert _test_args(chebyshevu(x, 2))

def test_sympy__functions__special__polynomials__chebyshevu_root():
    if False:
        i = 10
        return i + 15
    from sympy.functions.special.polynomials import chebyshevu_root
    assert _test_args(chebyshevu_root(3, 2))

def test_sympy__functions__special__polynomials__hermite():
    if False:
        return 10
    from sympy.functions.special.polynomials import hermite
    assert _test_args(hermite(x, 2))

def test_sympy__functions__special__polynomials__hermite_prob():
    if False:
        return 10
    from sympy.functions.special.polynomials import hermite_prob
    assert _test_args(hermite_prob(x, 2))

def test_sympy__functions__special__polynomials__legendre():
    if False:
        i = 10
        return i + 15
    from sympy.functions.special.polynomials import legendre
    assert _test_args(legendre(x, 2))

def test_sympy__functions__special__polynomials__assoc_legendre():
    if False:
        print('Hello World!')
    from sympy.functions.special.polynomials import assoc_legendre
    assert _test_args(assoc_legendre(x, 0, y))

def test_sympy__functions__special__polynomials__laguerre():
    if False:
        i = 10
        return i + 15
    from sympy.functions.special.polynomials import laguerre
    assert _test_args(laguerre(x, 2))

def test_sympy__functions__special__polynomials__assoc_laguerre():
    if False:
        for i in range(10):
            print('nop')
    from sympy.functions.special.polynomials import assoc_laguerre
    assert _test_args(assoc_laguerre(x, 0, y))

def test_sympy__functions__special__spherical_harmonics__Ynm():
    if False:
        print('Hello World!')
    from sympy.functions.special.spherical_harmonics import Ynm
    assert _test_args(Ynm(1, 1, x, y))

def test_sympy__functions__special__spherical_harmonics__Znm():
    if False:
        return 10
    from sympy.functions.special.spherical_harmonics import Znm
    assert _test_args(Znm(x, y, 1, 1))

def test_sympy__functions__special__tensor_functions__LeviCivita():
    if False:
        i = 10
        return i + 15
    from sympy.functions.special.tensor_functions import LeviCivita
    assert _test_args(LeviCivita(x, y, 2))

def test_sympy__functions__special__tensor_functions__KroneckerDelta():
    if False:
        while True:
            i = 10
    from sympy.functions.special.tensor_functions import KroneckerDelta
    assert _test_args(KroneckerDelta(x, y))

def test_sympy__functions__special__zeta_functions__dirichlet_eta():
    if False:
        i = 10
        return i + 15
    from sympy.functions.special.zeta_functions import dirichlet_eta
    assert _test_args(dirichlet_eta(x))

def test_sympy__functions__special__zeta_functions__riemann_xi():
    if False:
        return 10
    from sympy.functions.special.zeta_functions import riemann_xi
    assert _test_args(riemann_xi(x))

def test_sympy__functions__special__zeta_functions__zeta():
    if False:
        print('Hello World!')
    from sympy.functions.special.zeta_functions import zeta
    assert _test_args(zeta(101))

def test_sympy__functions__special__zeta_functions__lerchphi():
    if False:
        i = 10
        return i + 15
    from sympy.functions.special.zeta_functions import lerchphi
    assert _test_args(lerchphi(x, y, z))

def test_sympy__functions__special__zeta_functions__polylog():
    if False:
        print('Hello World!')
    from sympy.functions.special.zeta_functions import polylog
    assert _test_args(polylog(x, y))

def test_sympy__functions__special__zeta_functions__stieltjes():
    if False:
        i = 10
        return i + 15
    from sympy.functions.special.zeta_functions import stieltjes
    assert _test_args(stieltjes(x, y))

def test_sympy__integrals__integrals__Integral():
    if False:
        while True:
            i = 10
    from sympy.integrals.integrals import Integral
    assert _test_args(Integral(2, (x, 0, 1)))

def test_sympy__integrals__risch__NonElementaryIntegral():
    if False:
        print('Hello World!')
    from sympy.integrals.risch import NonElementaryIntegral
    assert _test_args(NonElementaryIntegral(exp(-x ** 2), x))

@SKIP('abstract class')
def test_sympy__integrals__transforms__IntegralTransform():
    if False:
        for i in range(10):
            print('nop')
    pass

def test_sympy__integrals__transforms__MellinTransform():
    if False:
        return 10
    from sympy.integrals.transforms import MellinTransform
    assert _test_args(MellinTransform(2, x, y))

def test_sympy__integrals__transforms__InverseMellinTransform():
    if False:
        return 10
    from sympy.integrals.transforms import InverseMellinTransform
    assert _test_args(InverseMellinTransform(2, x, y, 0, 1))

def test_sympy__integrals__laplace__LaplaceTransform():
    if False:
        for i in range(10):
            print('nop')
    from sympy.integrals.laplace import LaplaceTransform
    assert _test_args(LaplaceTransform(2, x, y))

def test_sympy__integrals__laplace__InverseLaplaceTransform():
    if False:
        print('Hello World!')
    from sympy.integrals.laplace import InverseLaplaceTransform
    assert _test_args(InverseLaplaceTransform(2, x, y, 0))

@SKIP('abstract class')
def test_sympy__integrals__transforms__FourierTypeTransform():
    if False:
        i = 10
        return i + 15
    pass

def test_sympy__integrals__transforms__InverseFourierTransform():
    if False:
        for i in range(10):
            print('nop')
    from sympy.integrals.transforms import InverseFourierTransform
    assert _test_args(InverseFourierTransform(2, x, y))

def test_sympy__integrals__transforms__FourierTransform():
    if False:
        i = 10
        return i + 15
    from sympy.integrals.transforms import FourierTransform
    assert _test_args(FourierTransform(2, x, y))

@SKIP('abstract class')
def test_sympy__integrals__transforms__SineCosineTypeTransform():
    if False:
        return 10
    pass

def test_sympy__integrals__transforms__InverseSineTransform():
    if False:
        print('Hello World!')
    from sympy.integrals.transforms import InverseSineTransform
    assert _test_args(InverseSineTransform(2, x, y))

def test_sympy__integrals__transforms__SineTransform():
    if False:
        print('Hello World!')
    from sympy.integrals.transforms import SineTransform
    assert _test_args(SineTransform(2, x, y))

def test_sympy__integrals__transforms__InverseCosineTransform():
    if False:
        for i in range(10):
            print('nop')
    from sympy.integrals.transforms import InverseCosineTransform
    assert _test_args(InverseCosineTransform(2, x, y))

def test_sympy__integrals__transforms__CosineTransform():
    if False:
        for i in range(10):
            print('nop')
    from sympy.integrals.transforms import CosineTransform
    assert _test_args(CosineTransform(2, x, y))

@SKIP('abstract class')
def test_sympy__integrals__transforms__HankelTypeTransform():
    if False:
        i = 10
        return i + 15
    pass

def test_sympy__integrals__transforms__InverseHankelTransform():
    if False:
        while True:
            i = 10
    from sympy.integrals.transforms import InverseHankelTransform
    assert _test_args(InverseHankelTransform(2, x, y, 0))

def test_sympy__integrals__transforms__HankelTransform():
    if False:
        i = 10
        return i + 15
    from sympy.integrals.transforms import HankelTransform
    assert _test_args(HankelTransform(2, x, y, 0))

def test_sympy__liealgebras__cartan_type__Standard_Cartan():
    if False:
        i = 10
        return i + 15
    from sympy.liealgebras.cartan_type import Standard_Cartan
    assert _test_args(Standard_Cartan('A', 2))

def test_sympy__liealgebras__weyl_group__WeylGroup():
    if False:
        while True:
            i = 10
    from sympy.liealgebras.weyl_group import WeylGroup
    assert _test_args(WeylGroup('B4'))

def test_sympy__liealgebras__root_system__RootSystem():
    if False:
        for i in range(10):
            print('nop')
    from sympy.liealgebras.root_system import RootSystem
    assert _test_args(RootSystem('A2'))

def test_sympy__liealgebras__type_a__TypeA():
    if False:
        return 10
    from sympy.liealgebras.type_a import TypeA
    assert _test_args(TypeA(2))

def test_sympy__liealgebras__type_b__TypeB():
    if False:
        print('Hello World!')
    from sympy.liealgebras.type_b import TypeB
    assert _test_args(TypeB(4))

def test_sympy__liealgebras__type_c__TypeC():
    if False:
        i = 10
        return i + 15
    from sympy.liealgebras.type_c import TypeC
    assert _test_args(TypeC(4))

def test_sympy__liealgebras__type_d__TypeD():
    if False:
        while True:
            i = 10
    from sympy.liealgebras.type_d import TypeD
    assert _test_args(TypeD(4))

def test_sympy__liealgebras__type_e__TypeE():
    if False:
        i = 10
        return i + 15
    from sympy.liealgebras.type_e import TypeE
    assert _test_args(TypeE(6))

def test_sympy__liealgebras__type_f__TypeF():
    if False:
        for i in range(10):
            print('nop')
    from sympy.liealgebras.type_f import TypeF
    assert _test_args(TypeF(4))

def test_sympy__liealgebras__type_g__TypeG():
    if False:
        for i in range(10):
            print('nop')
    from sympy.liealgebras.type_g import TypeG
    assert _test_args(TypeG(2))

def test_sympy__logic__boolalg__And():
    if False:
        return 10
    from sympy.logic.boolalg import And
    assert _test_args(And(x, y, 1))

@SKIP('abstract class')
def test_sympy__logic__boolalg__Boolean():
    if False:
        i = 10
        return i + 15
    pass

def test_sympy__logic__boolalg__BooleanFunction():
    if False:
        while True:
            i = 10
    from sympy.logic.boolalg import BooleanFunction
    assert _test_args(BooleanFunction(1, 2, 3))

@SKIP('abstract class')
def test_sympy__logic__boolalg__BooleanAtom():
    if False:
        i = 10
        return i + 15
    pass

def test_sympy__logic__boolalg__BooleanTrue():
    if False:
        print('Hello World!')
    from sympy.logic.boolalg import true
    assert _test_args(true)

def test_sympy__logic__boolalg__BooleanFalse():
    if False:
        for i in range(10):
            print('nop')
    from sympy.logic.boolalg import false
    assert _test_args(false)

def test_sympy__logic__boolalg__Equivalent():
    if False:
        return 10
    from sympy.logic.boolalg import Equivalent
    assert _test_args(Equivalent(x, 2))

def test_sympy__logic__boolalg__ITE():
    if False:
        i = 10
        return i + 15
    from sympy.logic.boolalg import ITE
    assert _test_args(ITE(x, y, 1))

def test_sympy__logic__boolalg__Implies():
    if False:
        i = 10
        return i + 15
    from sympy.logic.boolalg import Implies
    assert _test_args(Implies(x, y))

def test_sympy__logic__boolalg__Nand():
    if False:
        while True:
            i = 10
    from sympy.logic.boolalg import Nand
    assert _test_args(Nand(x, y, 1))

def test_sympy__logic__boolalg__Nor():
    if False:
        for i in range(10):
            print('nop')
    from sympy.logic.boolalg import Nor
    assert _test_args(Nor(x, y))

def test_sympy__logic__boolalg__Not():
    if False:
        for i in range(10):
            print('nop')
    from sympy.logic.boolalg import Not
    assert _test_args(Not(x))

def test_sympy__logic__boolalg__Or():
    if False:
        while True:
            i = 10
    from sympy.logic.boolalg import Or
    assert _test_args(Or(x, y))

def test_sympy__logic__boolalg__Xor():
    if False:
        for i in range(10):
            print('nop')
    from sympy.logic.boolalg import Xor
    assert _test_args(Xor(x, y, 2))

def test_sympy__logic__boolalg__Xnor():
    if False:
        while True:
            i = 10
    from sympy.logic.boolalg import Xnor
    assert _test_args(Xnor(x, y, 2))

def test_sympy__logic__boolalg__Exclusive():
    if False:
        while True:
            i = 10
    from sympy.logic.boolalg import Exclusive
    assert _test_args(Exclusive(x, y, z))

def test_sympy__matrices__matrices__DeferredVector():
    if False:
        print('Hello World!')
    from sympy.matrices.matrices import DeferredVector
    assert _test_args(DeferredVector('X'))

@SKIP('abstract class')
def test_sympy__matrices__expressions__matexpr__MatrixBase():
    if False:
        i = 10
        return i + 15
    pass

@SKIP('abstract class')
def test_sympy__matrices__immutable__ImmutableRepMatrix():
    if False:
        i = 10
        return i + 15
    pass

def test_sympy__matrices__immutable__ImmutableDenseMatrix():
    if False:
        i = 10
        return i + 15
    from sympy.matrices.immutable import ImmutableDenseMatrix
    m = ImmutableDenseMatrix([[1, 2], [3, 4]])
    assert _test_args(m)
    assert _test_args(Basic(*list(m)))
    m = ImmutableDenseMatrix(1, 1, [1])
    assert _test_args(m)
    assert _test_args(Basic(*list(m)))
    m = ImmutableDenseMatrix(2, 2, lambda i, j: 1)
    assert m[0, 0] is S.One
    m = ImmutableDenseMatrix(2, 2, lambda i, j: 1 / (1 + i) + 1 / (1 + j))
    assert m[1, 1] is S.One
    assert _test_args(m)
    assert _test_args(Basic(*list(m)))

def test_sympy__matrices__immutable__ImmutableSparseMatrix():
    if False:
        return 10
    from sympy.matrices.immutable import ImmutableSparseMatrix
    m = ImmutableSparseMatrix([[1, 2], [3, 4]])
    assert _test_args(m)
    assert _test_args(Basic(*list(m)))
    m = ImmutableSparseMatrix(1, 1, {(0, 0): 1})
    assert _test_args(m)
    assert _test_args(Basic(*list(m)))
    m = ImmutableSparseMatrix(1, 1, [1])
    assert _test_args(m)
    assert _test_args(Basic(*list(m)))
    m = ImmutableSparseMatrix(2, 2, lambda i, j: 1)
    assert m[0, 0] is S.One
    m = ImmutableSparseMatrix(2, 2, lambda i, j: 1 / (1 + i) + 1 / (1 + j))
    assert m[1, 1] is S.One
    assert _test_args(m)
    assert _test_args(Basic(*list(m)))

def test_sympy__matrices__expressions__slice__MatrixSlice():
    if False:
        return 10
    from sympy.matrices.expressions.slice import MatrixSlice
    from sympy.matrices.expressions import MatrixSymbol
    X = MatrixSymbol('X', 4, 4)
    assert _test_args(MatrixSlice(X, (0, 2), (0, 2)))

def test_sympy__matrices__expressions__applyfunc__ElementwiseApplyFunction():
    if False:
        while True:
            i = 10
    from sympy.matrices.expressions.applyfunc import ElementwiseApplyFunction
    from sympy.matrices.expressions import MatrixSymbol
    X = MatrixSymbol('X', x, x)
    func = Lambda(x, x ** 2)
    assert _test_args(ElementwiseApplyFunction(func, X))

def test_sympy__matrices__expressions__blockmatrix__BlockDiagMatrix():
    if False:
        i = 10
        return i + 15
    from sympy.matrices.expressions.blockmatrix import BlockDiagMatrix
    from sympy.matrices.expressions import MatrixSymbol
    X = MatrixSymbol('X', x, x)
    Y = MatrixSymbol('Y', y, y)
    assert _test_args(BlockDiagMatrix(X, Y))

def test_sympy__matrices__expressions__blockmatrix__BlockMatrix():
    if False:
        for i in range(10):
            print('nop')
    from sympy.matrices.expressions.blockmatrix import BlockMatrix
    from sympy.matrices.expressions import MatrixSymbol, ZeroMatrix
    X = MatrixSymbol('X', x, x)
    Y = MatrixSymbol('Y', y, y)
    Z = MatrixSymbol('Z', x, y)
    O = ZeroMatrix(y, x)
    assert _test_args(BlockMatrix([[X, Z], [O, Y]]))

def test_sympy__matrices__expressions__inverse__Inverse():
    if False:
        return 10
    from sympy.matrices.expressions.inverse import Inverse
    from sympy.matrices.expressions import MatrixSymbol
    assert _test_args(Inverse(MatrixSymbol('A', 3, 3)))

def test_sympy__matrices__expressions__matadd__MatAdd():
    if False:
        return 10
    from sympy.matrices.expressions.matadd import MatAdd
    from sympy.matrices.expressions import MatrixSymbol
    X = MatrixSymbol('X', x, y)
    Y = MatrixSymbol('Y', x, y)
    assert _test_args(MatAdd(X, Y))

@SKIP('abstract class')
def test_sympy__matrices__expressions__matexpr__MatrixExpr():
    if False:
        i = 10
        return i + 15
    pass

def test_sympy__matrices__expressions__matexpr__MatrixElement():
    if False:
        i = 10
        return i + 15
    from sympy.matrices.expressions.matexpr import MatrixSymbol, MatrixElement
    from sympy.core.singleton import S
    assert _test_args(MatrixElement(MatrixSymbol('A', 3, 5), S(2), S(3)))

def test_sympy__matrices__expressions__matexpr__MatrixSymbol():
    if False:
        i = 10
        return i + 15
    from sympy.matrices.expressions.matexpr import MatrixSymbol
    assert _test_args(MatrixSymbol('A', 3, 5))

def test_sympy__matrices__expressions__special__OneMatrix():
    if False:
        print('Hello World!')
    from sympy.matrices.expressions.special import OneMatrix
    assert _test_args(OneMatrix(3, 5))

def test_sympy__matrices__expressions__special__ZeroMatrix():
    if False:
        print('Hello World!')
    from sympy.matrices.expressions.special import ZeroMatrix
    assert _test_args(ZeroMatrix(3, 5))

def test_sympy__matrices__expressions__special__GenericZeroMatrix():
    if False:
        print('Hello World!')
    from sympy.matrices.expressions.special import GenericZeroMatrix
    assert _test_args(GenericZeroMatrix())

def test_sympy__matrices__expressions__special__Identity():
    if False:
        for i in range(10):
            print('nop')
    from sympy.matrices.expressions.special import Identity
    assert _test_args(Identity(3))

def test_sympy__matrices__expressions__special__GenericIdentity():
    if False:
        print('Hello World!')
    from sympy.matrices.expressions.special import GenericIdentity
    assert _test_args(GenericIdentity())

def test_sympy__matrices__expressions__sets__MatrixSet():
    if False:
        return 10
    from sympy.matrices.expressions.sets import MatrixSet
    from sympy.core.singleton import S
    assert _test_args(MatrixSet(2, 2, S.Reals))

def test_sympy__matrices__expressions__matmul__MatMul():
    if False:
        print('Hello World!')
    from sympy.matrices.expressions.matmul import MatMul
    from sympy.matrices.expressions import MatrixSymbol
    X = MatrixSymbol('X', x, y)
    Y = MatrixSymbol('Y', y, x)
    assert _test_args(MatMul(X, Y))

def test_sympy__matrices__expressions__dotproduct__DotProduct():
    if False:
        while True:
            i = 10
    from sympy.matrices.expressions.dotproduct import DotProduct
    from sympy.matrices.expressions import MatrixSymbol
    X = MatrixSymbol('X', x, 1)
    Y = MatrixSymbol('Y', x, 1)
    assert _test_args(DotProduct(X, Y))

def test_sympy__matrices__expressions__diagonal__DiagonalMatrix():
    if False:
        for i in range(10):
            print('nop')
    from sympy.matrices.expressions.diagonal import DiagonalMatrix
    from sympy.matrices.expressions import MatrixSymbol
    x = MatrixSymbol('x', 10, 1)
    assert _test_args(DiagonalMatrix(x))

def test_sympy__matrices__expressions__diagonal__DiagonalOf():
    if False:
        for i in range(10):
            print('nop')
    from sympy.matrices.expressions.diagonal import DiagonalOf
    from sympy.matrices.expressions import MatrixSymbol
    X = MatrixSymbol('x', 10, 10)
    assert _test_args(DiagonalOf(X))

def test_sympy__matrices__expressions__diagonal__DiagMatrix():
    if False:
        for i in range(10):
            print('nop')
    from sympy.matrices.expressions.diagonal import DiagMatrix
    from sympy.matrices.expressions import MatrixSymbol
    x = MatrixSymbol('x', 10, 1)
    assert _test_args(DiagMatrix(x))

def test_sympy__matrices__expressions__hadamard__HadamardProduct():
    if False:
        i = 10
        return i + 15
    from sympy.matrices.expressions.hadamard import HadamardProduct
    from sympy.matrices.expressions import MatrixSymbol
    X = MatrixSymbol('X', x, y)
    Y = MatrixSymbol('Y', x, y)
    assert _test_args(HadamardProduct(X, Y))

def test_sympy__matrices__expressions__hadamard__HadamardPower():
    if False:
        for i in range(10):
            print('nop')
    from sympy.matrices.expressions.hadamard import HadamardPower
    from sympy.matrices.expressions import MatrixSymbol
    from sympy.core.symbol import Symbol
    X = MatrixSymbol('X', x, y)
    n = Symbol('n')
    assert _test_args(HadamardPower(X, n))

def test_sympy__matrices__expressions__kronecker__KroneckerProduct():
    if False:
        for i in range(10):
            print('nop')
    from sympy.matrices.expressions.kronecker import KroneckerProduct
    from sympy.matrices.expressions import MatrixSymbol
    X = MatrixSymbol('X', x, y)
    Y = MatrixSymbol('Y', x, y)
    assert _test_args(KroneckerProduct(X, Y))

def test_sympy__matrices__expressions__matpow__MatPow():
    if False:
        while True:
            i = 10
    from sympy.matrices.expressions.matpow import MatPow
    from sympy.matrices.expressions import MatrixSymbol
    X = MatrixSymbol('X', x, x)
    assert _test_args(MatPow(X, 2))

def test_sympy__matrices__expressions__transpose__Transpose():
    if False:
        i = 10
        return i + 15
    from sympy.matrices.expressions.transpose import Transpose
    from sympy.matrices.expressions import MatrixSymbol
    assert _test_args(Transpose(MatrixSymbol('A', 3, 5)))

def test_sympy__matrices__expressions__adjoint__Adjoint():
    if False:
        print('Hello World!')
    from sympy.matrices.expressions.adjoint import Adjoint
    from sympy.matrices.expressions import MatrixSymbol
    assert _test_args(Adjoint(MatrixSymbol('A', 3, 5)))

def test_sympy__matrices__expressions__trace__Trace():
    if False:
        return 10
    from sympy.matrices.expressions.trace import Trace
    from sympy.matrices.expressions import MatrixSymbol
    assert _test_args(Trace(MatrixSymbol('A', 3, 3)))

def test_sympy__matrices__expressions__determinant__Determinant():
    if False:
        print('Hello World!')
    from sympy.matrices.expressions.determinant import Determinant
    from sympy.matrices.expressions import MatrixSymbol
    assert _test_args(Determinant(MatrixSymbol('A', 3, 3)))

def test_sympy__matrices__expressions__determinant__Permanent():
    if False:
        i = 10
        return i + 15
    from sympy.matrices.expressions.determinant import Permanent
    from sympy.matrices.expressions import MatrixSymbol
    assert _test_args(Permanent(MatrixSymbol('A', 3, 4)))

def test_sympy__matrices__expressions__funcmatrix__FunctionMatrix():
    if False:
        while True:
            i = 10
    from sympy.matrices.expressions.funcmatrix import FunctionMatrix
    from sympy.core.symbol import symbols
    (i, j) = symbols('i,j')
    assert _test_args(FunctionMatrix(3, 3, Lambda((i, j), i - j)))

def test_sympy__matrices__expressions__fourier__DFT():
    if False:
        i = 10
        return i + 15
    from sympy.matrices.expressions.fourier import DFT
    from sympy.core.singleton import S
    assert _test_args(DFT(S(2)))

def test_sympy__matrices__expressions__fourier__IDFT():
    if False:
        for i in range(10):
            print('nop')
    from sympy.matrices.expressions.fourier import IDFT
    from sympy.core.singleton import S
    assert _test_args(IDFT(S(2)))
from sympy.matrices.expressions import MatrixSymbol
X = MatrixSymbol('X', 10, 10)

def test_sympy__matrices__expressions__factorizations__LofLU():
    if False:
        print('Hello World!')
    from sympy.matrices.expressions.factorizations import LofLU
    assert _test_args(LofLU(X))

def test_sympy__matrices__expressions__factorizations__UofLU():
    if False:
        while True:
            i = 10
    from sympy.matrices.expressions.factorizations import UofLU
    assert _test_args(UofLU(X))

def test_sympy__matrices__expressions__factorizations__QofQR():
    if False:
        while True:
            i = 10
    from sympy.matrices.expressions.factorizations import QofQR
    assert _test_args(QofQR(X))

def test_sympy__matrices__expressions__factorizations__RofQR():
    if False:
        for i in range(10):
            print('nop')
    from sympy.matrices.expressions.factorizations import RofQR
    assert _test_args(RofQR(X))

def test_sympy__matrices__expressions__factorizations__LofCholesky():
    if False:
        i = 10
        return i + 15
    from sympy.matrices.expressions.factorizations import LofCholesky
    assert _test_args(LofCholesky(X))

def test_sympy__matrices__expressions__factorizations__UofCholesky():
    if False:
        i = 10
        return i + 15
    from sympy.matrices.expressions.factorizations import UofCholesky
    assert _test_args(UofCholesky(X))

def test_sympy__matrices__expressions__factorizations__EigenVectors():
    if False:
        for i in range(10):
            print('nop')
    from sympy.matrices.expressions.factorizations import EigenVectors
    assert _test_args(EigenVectors(X))

def test_sympy__matrices__expressions__factorizations__EigenValues():
    if False:
        return 10
    from sympy.matrices.expressions.factorizations import EigenValues
    assert _test_args(EigenValues(X))

def test_sympy__matrices__expressions__factorizations__UofSVD():
    if False:
        for i in range(10):
            print('nop')
    from sympy.matrices.expressions.factorizations import UofSVD
    assert _test_args(UofSVD(X))

def test_sympy__matrices__expressions__factorizations__VofSVD():
    if False:
        print('Hello World!')
    from sympy.matrices.expressions.factorizations import VofSVD
    assert _test_args(VofSVD(X))

def test_sympy__matrices__expressions__factorizations__SofSVD():
    if False:
        while True:
            i = 10
    from sympy.matrices.expressions.factorizations import SofSVD
    assert _test_args(SofSVD(X))

@SKIP('abstract class')
def test_sympy__matrices__expressions__factorizations__Factorization():
    if False:
        i = 10
        return i + 15
    pass

def test_sympy__matrices__expressions__permutation__PermutationMatrix():
    if False:
        i = 10
        return i + 15
    from sympy.combinatorics import Permutation
    from sympy.matrices.expressions.permutation import PermutationMatrix
    assert _test_args(PermutationMatrix(Permutation([2, 0, 1])))

def test_sympy__matrices__expressions__permutation__MatrixPermute():
    if False:
        return 10
    from sympy.combinatorics import Permutation
    from sympy.matrices.expressions.matexpr import MatrixSymbol
    from sympy.matrices.expressions.permutation import MatrixPermute
    A = MatrixSymbol('A', 3, 3)
    assert _test_args(MatrixPermute(A, Permutation([2, 0, 1])))

def test_sympy__matrices__expressions__companion__CompanionMatrix():
    if False:
        for i in range(10):
            print('nop')
    from sympy.core.symbol import Symbol
    from sympy.matrices.expressions.companion import CompanionMatrix
    from sympy.polys.polytools import Poly
    x = Symbol('x')
    p = Poly([1, 2, 3], x)
    assert _test_args(CompanionMatrix(p))

def test_sympy__physics__vector__frame__CoordinateSym():
    if False:
        for i in range(10):
            print('nop')
    from sympy.physics.vector import CoordinateSym
    from sympy.physics.vector import ReferenceFrame
    assert _test_args(CoordinateSym('R_x', ReferenceFrame('R'), 0))

@SKIP('abstract class')
def test_sympy__physics__biomechanics__curve__CharacteristicCurveFunction():
    if False:
        for i in range(10):
            print('nop')
    pass

def test_sympy__physics__biomechanics__curve__TendonForceLengthDeGroote2016():
    if False:
        print('Hello World!')
    from sympy.physics.biomechanics import TendonForceLengthDeGroote2016
    (l_T_tilde, c0, c1, c2, c3) = symbols('l_T_tilde, c0, c1, c2, c3')
    assert _test_args(TendonForceLengthDeGroote2016(l_T_tilde, c0, c1, c2, c3))

def test_sympy__physics__biomechanics__curve__TendonForceLengthInverseDeGroote2016():
    if False:
        i = 10
        return i + 15
    from sympy.physics.biomechanics import TendonForceLengthInverseDeGroote2016
    (fl_T, c0, c1, c2, c3) = symbols('fl_T, c0, c1, c2, c3')
    assert _test_args(TendonForceLengthInverseDeGroote2016(fl_T, c0, c1, c2, c3))

def test_sympy__physics__biomechanics__curve__FiberForceLengthPassiveDeGroote2016():
    if False:
        i = 10
        return i + 15
    from sympy.physics.biomechanics import FiberForceLengthPassiveDeGroote2016
    (l_M_tilde, c0, c1) = symbols('l_M_tilde, c0, c1')
    assert _test_args(FiberForceLengthPassiveDeGroote2016(l_M_tilde, c0, c1))

def test_sympy__physics__biomechanics__curve__FiberForceLengthPassiveInverseDeGroote2016():
    if False:
        i = 10
        return i + 15
    from sympy.physics.biomechanics import FiberForceLengthPassiveInverseDeGroote2016
    (fl_M_pas, c0, c1) = symbols('fl_M_pas, c0, c1')
    assert _test_args(FiberForceLengthPassiveInverseDeGroote2016(fl_M_pas, c0, c1))

def test_sympy__physics__biomechanics__curve__FiberForceLengthActiveDeGroote2016():
    if False:
        i = 10
        return i + 15
    from sympy.physics.biomechanics import FiberForceLengthActiveDeGroote2016
    (l_M_tilde, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11) = symbols('l_M_tilde, c0:12')
    assert _test_args(FiberForceLengthActiveDeGroote2016(l_M_tilde, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11))

def test_sympy__physics__biomechanics__curve__FiberForceVelocityDeGroote2016():
    if False:
        i = 10
        return i + 15
    from sympy.physics.biomechanics import FiberForceVelocityDeGroote2016
    (v_M_tilde, c0, c1, c2, c3) = symbols('v_M_tilde, c0, c1, c2, c3')
    assert _test_args(FiberForceVelocityDeGroote2016(v_M_tilde, c0, c1, c2, c3))

def test_sympy__physics__biomechanics__curve__FiberForceVelocityInverseDeGroote2016():
    if False:
        while True:
            i = 10
    from sympy.physics.biomechanics import FiberForceVelocityInverseDeGroote2016
    (fv_M, c0, c1, c2, c3) = symbols('fv_M, c0, c1, c2, c3')
    assert _test_args(FiberForceVelocityInverseDeGroote2016(fv_M, c0, c1, c2, c3))

def test_sympy__physics__paulialgebra__Pauli():
    if False:
        while True:
            i = 10
    from sympy.physics.paulialgebra import Pauli
    assert _test_args(Pauli(1))

def test_sympy__physics__quantum__anticommutator__AntiCommutator():
    if False:
        while True:
            i = 10
    from sympy.physics.quantum.anticommutator import AntiCommutator
    assert _test_args(AntiCommutator(x, y))

def test_sympy__physics__quantum__cartesian__PositionBra3D():
    if False:
        i = 10
        return i + 15
    from sympy.physics.quantum.cartesian import PositionBra3D
    assert _test_args(PositionBra3D(x, y, z))

def test_sympy__physics__quantum__cartesian__PositionKet3D():
    if False:
        for i in range(10):
            print('nop')
    from sympy.physics.quantum.cartesian import PositionKet3D
    assert _test_args(PositionKet3D(x, y, z))

def test_sympy__physics__quantum__cartesian__PositionState3D():
    if False:
        i = 10
        return i + 15
    from sympy.physics.quantum.cartesian import PositionState3D
    assert _test_args(PositionState3D(x, y, z))

def test_sympy__physics__quantum__cartesian__PxBra():
    if False:
        return 10
    from sympy.physics.quantum.cartesian import PxBra
    assert _test_args(PxBra(x, y, z))

def test_sympy__physics__quantum__cartesian__PxKet():
    if False:
        return 10
    from sympy.physics.quantum.cartesian import PxKet
    assert _test_args(PxKet(x, y, z))

def test_sympy__physics__quantum__cartesian__PxOp():
    if False:
        for i in range(10):
            print('nop')
    from sympy.physics.quantum.cartesian import PxOp
    assert _test_args(PxOp(x, y, z))

def test_sympy__physics__quantum__cartesian__XBra():
    if False:
        i = 10
        return i + 15
    from sympy.physics.quantum.cartesian import XBra
    assert _test_args(XBra(x))

def test_sympy__physics__quantum__cartesian__XKet():
    if False:
        return 10
    from sympy.physics.quantum.cartesian import XKet
    assert _test_args(XKet(x))

def test_sympy__physics__quantum__cartesian__XOp():
    if False:
        print('Hello World!')
    from sympy.physics.quantum.cartesian import XOp
    assert _test_args(XOp(x))

def test_sympy__physics__quantum__cartesian__YOp():
    if False:
        i = 10
        return i + 15
    from sympy.physics.quantum.cartesian import YOp
    assert _test_args(YOp(x))

def test_sympy__physics__quantum__cartesian__ZOp():
    if False:
        while True:
            i = 10
    from sympy.physics.quantum.cartesian import ZOp
    assert _test_args(ZOp(x))

def test_sympy__physics__quantum__cg__CG():
    if False:
        while True:
            i = 10
    from sympy.physics.quantum.cg import CG
    from sympy.core.singleton import S
    assert _test_args(CG(Rational(3, 2), Rational(3, 2), S.Half, Rational(-1, 2), 1, 1))

def test_sympy__physics__quantum__cg__Wigner3j():
    if False:
        for i in range(10):
            print('nop')
    from sympy.physics.quantum.cg import Wigner3j
    assert _test_args(Wigner3j(6, 0, 4, 0, 2, 0))

def test_sympy__physics__quantum__cg__Wigner6j():
    if False:
        print('Hello World!')
    from sympy.physics.quantum.cg import Wigner6j
    assert _test_args(Wigner6j(1, 2, 3, 2, 1, 2))

def test_sympy__physics__quantum__cg__Wigner9j():
    if False:
        for i in range(10):
            print('nop')
    from sympy.physics.quantum.cg import Wigner9j
    assert _test_args(Wigner9j(2, 1, 1, Rational(3, 2), S.Half, 1, S.Half, S.Half, 0))

def test_sympy__physics__quantum__circuitplot__Mz():
    if False:
        while True:
            i = 10
    from sympy.physics.quantum.circuitplot import Mz
    assert _test_args(Mz(0))

def test_sympy__physics__quantum__circuitplot__Mx():
    if False:
        i = 10
        return i + 15
    from sympy.physics.quantum.circuitplot import Mx
    assert _test_args(Mx(0))

def test_sympy__physics__quantum__commutator__Commutator():
    if False:
        while True:
            i = 10
    from sympy.physics.quantum.commutator import Commutator
    (A, B) = symbols('A,B', commutative=False)
    assert _test_args(Commutator(A, B))

def test_sympy__physics__quantum__constants__HBar():
    if False:
        i = 10
        return i + 15
    from sympy.physics.quantum.constants import HBar
    assert _test_args(HBar())

def test_sympy__physics__quantum__dagger__Dagger():
    if False:
        return 10
    from sympy.physics.quantum.dagger import Dagger
    from sympy.physics.quantum.state import Ket
    assert _test_args(Dagger(Dagger(Ket('psi'))))

def test_sympy__physics__quantum__gate__CGate():
    if False:
        while True:
            i = 10
    from sympy.physics.quantum.gate import CGate, Gate
    assert _test_args(CGate((0, 1), Gate(2)))

def test_sympy__physics__quantum__gate__CGateS():
    if False:
        return 10
    from sympy.physics.quantum.gate import CGateS, Gate
    assert _test_args(CGateS((0, 1), Gate(2)))

def test_sympy__physics__quantum__gate__CNotGate():
    if False:
        i = 10
        return i + 15
    from sympy.physics.quantum.gate import CNotGate
    assert _test_args(CNotGate(0, 1))

def test_sympy__physics__quantum__gate__Gate():
    if False:
        i = 10
        return i + 15
    from sympy.physics.quantum.gate import Gate
    assert _test_args(Gate(0))

def test_sympy__physics__quantum__gate__HadamardGate():
    if False:
        for i in range(10):
            print('nop')
    from sympy.physics.quantum.gate import HadamardGate
    assert _test_args(HadamardGate(0))

def test_sympy__physics__quantum__gate__IdentityGate():
    if False:
        while True:
            i = 10
    from sympy.physics.quantum.gate import IdentityGate
    assert _test_args(IdentityGate(0))

def test_sympy__physics__quantum__gate__OneQubitGate():
    if False:
        while True:
            i = 10
    from sympy.physics.quantum.gate import OneQubitGate
    assert _test_args(OneQubitGate(0))

def test_sympy__physics__quantum__gate__PhaseGate():
    if False:
        return 10
    from sympy.physics.quantum.gate import PhaseGate
    assert _test_args(PhaseGate(0))

def test_sympy__physics__quantum__gate__SwapGate():
    if False:
        return 10
    from sympy.physics.quantum.gate import SwapGate
    assert _test_args(SwapGate(0, 1))

def test_sympy__physics__quantum__gate__TGate():
    if False:
        while True:
            i = 10
    from sympy.physics.quantum.gate import TGate
    assert _test_args(TGate(0))

def test_sympy__physics__quantum__gate__TwoQubitGate():
    if False:
        while True:
            i = 10
    from sympy.physics.quantum.gate import TwoQubitGate
    assert _test_args(TwoQubitGate(0))

def test_sympy__physics__quantum__gate__UGate():
    if False:
        for i in range(10):
            print('nop')
    from sympy.physics.quantum.gate import UGate
    from sympy.matrices.immutable import ImmutableDenseMatrix
    from sympy.core.containers import Tuple
    from sympy.core.numbers import Integer
    assert _test_args(UGate(Tuple(Integer(1)), ImmutableDenseMatrix([[1, 0], [0, 2]])))

def test_sympy__physics__quantum__gate__XGate():
    if False:
        while True:
            i = 10
    from sympy.physics.quantum.gate import XGate
    assert _test_args(XGate(0))

def test_sympy__physics__quantum__gate__YGate():
    if False:
        while True:
            i = 10
    from sympy.physics.quantum.gate import YGate
    assert _test_args(YGate(0))

def test_sympy__physics__quantum__gate__ZGate():
    if False:
        i = 10
        return i + 15
    from sympy.physics.quantum.gate import ZGate
    assert _test_args(ZGate(0))

def test_sympy__physics__quantum__grover__OracleGateFunction():
    if False:
        return 10
    from sympy.physics.quantum.grover import OracleGateFunction

    @OracleGateFunction
    def f(qubit):
        if False:
            i = 10
            return i + 15
        return
    assert _test_args(f)

def test_sympy__physics__quantum__grover__OracleGate():
    if False:
        while True:
            i = 10
    from sympy.physics.quantum.grover import OracleGate

    def f(qubit):
        if False:
            return 10
        return
    assert _test_args(OracleGate(1, f))

def test_sympy__physics__quantum__grover__WGate():
    if False:
        i = 10
        return i + 15
    from sympy.physics.quantum.grover import WGate
    assert _test_args(WGate(1))

def test_sympy__physics__quantum__hilbert__ComplexSpace():
    if False:
        print('Hello World!')
    from sympy.physics.quantum.hilbert import ComplexSpace
    assert _test_args(ComplexSpace(x))

def test_sympy__physics__quantum__hilbert__DirectSumHilbertSpace():
    if False:
        while True:
            i = 10
    from sympy.physics.quantum.hilbert import DirectSumHilbertSpace, ComplexSpace, FockSpace
    c = ComplexSpace(2)
    f = FockSpace()
    assert _test_args(DirectSumHilbertSpace(c, f))

def test_sympy__physics__quantum__hilbert__FockSpace():
    if False:
        i = 10
        return i + 15
    from sympy.physics.quantum.hilbert import FockSpace
    assert _test_args(FockSpace())

def test_sympy__physics__quantum__hilbert__HilbertSpace():
    if False:
        for i in range(10):
            print('nop')
    from sympy.physics.quantum.hilbert import HilbertSpace
    assert _test_args(HilbertSpace())

def test_sympy__physics__quantum__hilbert__L2():
    if False:
        while True:
            i = 10
    from sympy.physics.quantum.hilbert import L2
    from sympy.core.numbers import oo
    from sympy.sets.sets import Interval
    assert _test_args(L2(Interval(0, oo)))

def test_sympy__physics__quantum__hilbert__TensorPowerHilbertSpace():
    if False:
        for i in range(10):
            print('nop')
    from sympy.physics.quantum.hilbert import TensorPowerHilbertSpace, FockSpace
    f = FockSpace()
    assert _test_args(TensorPowerHilbertSpace(f, 2))

def test_sympy__physics__quantum__hilbert__TensorProductHilbertSpace():
    if False:
        return 10
    from sympy.physics.quantum.hilbert import TensorProductHilbertSpace, FockSpace, ComplexSpace
    c = ComplexSpace(2)
    f = FockSpace()
    assert _test_args(TensorProductHilbertSpace(f, c))

def test_sympy__physics__quantum__innerproduct__InnerProduct():
    if False:
        for i in range(10):
            print('nop')
    from sympy.physics.quantum import Bra, Ket, InnerProduct
    b = Bra('b')
    k = Ket('k')
    assert _test_args(InnerProduct(b, k))

def test_sympy__physics__quantum__operator__DifferentialOperator():
    if False:
        for i in range(10):
            print('nop')
    from sympy.physics.quantum.operator import DifferentialOperator
    from sympy.core.function import Derivative, Function
    f = Function('f')
    assert _test_args(DifferentialOperator(1 / x * Derivative(f(x), x), f(x)))

def test_sympy__physics__quantum__operator__HermitianOperator():
    if False:
        for i in range(10):
            print('nop')
    from sympy.physics.quantum.operator import HermitianOperator
    assert _test_args(HermitianOperator('H'))

def test_sympy__physics__quantum__operator__IdentityOperator():
    if False:
        i = 10
        return i + 15
    from sympy.physics.quantum.operator import IdentityOperator
    assert _test_args(IdentityOperator(5))

def test_sympy__physics__quantum__operator__Operator():
    if False:
        return 10
    from sympy.physics.quantum.operator import Operator
    assert _test_args(Operator('A'))

def test_sympy__physics__quantum__operator__OuterProduct():
    if False:
        for i in range(10):
            print('nop')
    from sympy.physics.quantum.operator import OuterProduct
    from sympy.physics.quantum import Ket, Bra
    b = Bra('b')
    k = Ket('k')
    assert _test_args(OuterProduct(k, b))

def test_sympy__physics__quantum__operator__UnitaryOperator():
    if False:
        i = 10
        return i + 15
    from sympy.physics.quantum.operator import UnitaryOperator
    assert _test_args(UnitaryOperator('U'))

def test_sympy__physics__quantum__piab__PIABBra():
    if False:
        while True:
            i = 10
    from sympy.physics.quantum.piab import PIABBra
    assert _test_args(PIABBra('B'))

def test_sympy__physics__quantum__boson__BosonOp():
    if False:
        for i in range(10):
            print('nop')
    from sympy.physics.quantum.boson import BosonOp
    assert _test_args(BosonOp('a'))
    assert _test_args(BosonOp('a', False))

def test_sympy__physics__quantum__boson__BosonFockKet():
    if False:
        return 10
    from sympy.physics.quantum.boson import BosonFockKet
    assert _test_args(BosonFockKet(1))

def test_sympy__physics__quantum__boson__BosonFockBra():
    if False:
        return 10
    from sympy.physics.quantum.boson import BosonFockBra
    assert _test_args(BosonFockBra(1))

def test_sympy__physics__quantum__boson__BosonCoherentKet():
    if False:
        while True:
            i = 10
    from sympy.physics.quantum.boson import BosonCoherentKet
    assert _test_args(BosonCoherentKet(1))

def test_sympy__physics__quantum__boson__BosonCoherentBra():
    if False:
        return 10
    from sympy.physics.quantum.boson import BosonCoherentBra
    assert _test_args(BosonCoherentBra(1))

def test_sympy__physics__quantum__fermion__FermionOp():
    if False:
        return 10
    from sympy.physics.quantum.fermion import FermionOp
    assert _test_args(FermionOp('c'))
    assert _test_args(FermionOp('c', False))

def test_sympy__physics__quantum__fermion__FermionFockKet():
    if False:
        print('Hello World!')
    from sympy.physics.quantum.fermion import FermionFockKet
    assert _test_args(FermionFockKet(1))

def test_sympy__physics__quantum__fermion__FermionFockBra():
    if False:
        i = 10
        return i + 15
    from sympy.physics.quantum.fermion import FermionFockBra
    assert _test_args(FermionFockBra(1))

def test_sympy__physics__quantum__pauli__SigmaOpBase():
    if False:
        i = 10
        return i + 15
    from sympy.physics.quantum.pauli import SigmaOpBase
    assert _test_args(SigmaOpBase())

def test_sympy__physics__quantum__pauli__SigmaX():
    if False:
        i = 10
        return i + 15
    from sympy.physics.quantum.pauli import SigmaX
    assert _test_args(SigmaX())

def test_sympy__physics__quantum__pauli__SigmaY():
    if False:
        for i in range(10):
            print('nop')
    from sympy.physics.quantum.pauli import SigmaY
    assert _test_args(SigmaY())

def test_sympy__physics__quantum__pauli__SigmaZ():
    if False:
        i = 10
        return i + 15
    from sympy.physics.quantum.pauli import SigmaZ
    assert _test_args(SigmaZ())

def test_sympy__physics__quantum__pauli__SigmaMinus():
    if False:
        while True:
            i = 10
    from sympy.physics.quantum.pauli import SigmaMinus
    assert _test_args(SigmaMinus())

def test_sympy__physics__quantum__pauli__SigmaPlus():
    if False:
        while True:
            i = 10
    from sympy.physics.quantum.pauli import SigmaPlus
    assert _test_args(SigmaPlus())

def test_sympy__physics__quantum__pauli__SigmaZKet():
    if False:
        while True:
            i = 10
    from sympy.physics.quantum.pauli import SigmaZKet
    assert _test_args(SigmaZKet(0))

def test_sympy__physics__quantum__pauli__SigmaZBra():
    if False:
        for i in range(10):
            print('nop')
    from sympy.physics.quantum.pauli import SigmaZBra
    assert _test_args(SigmaZBra(0))

def test_sympy__physics__quantum__piab__PIABHamiltonian():
    if False:
        for i in range(10):
            print('nop')
    from sympy.physics.quantum.piab import PIABHamiltonian
    assert _test_args(PIABHamiltonian('P'))

def test_sympy__physics__quantum__piab__PIABKet():
    if False:
        i = 10
        return i + 15
    from sympy.physics.quantum.piab import PIABKet
    assert _test_args(PIABKet('K'))

def test_sympy__physics__quantum__qexpr__QExpr():
    if False:
        i = 10
        return i + 15
    from sympy.physics.quantum.qexpr import QExpr
    assert _test_args(QExpr(0))

def test_sympy__physics__quantum__qft__Fourier():
    if False:
        print('Hello World!')
    from sympy.physics.quantum.qft import Fourier
    assert _test_args(Fourier(0, 1))

def test_sympy__physics__quantum__qft__IQFT():
    if False:
        while True:
            i = 10
    from sympy.physics.quantum.qft import IQFT
    assert _test_args(IQFT(0, 1))

def test_sympy__physics__quantum__qft__QFT():
    if False:
        i = 10
        return i + 15
    from sympy.physics.quantum.qft import QFT
    assert _test_args(QFT(0, 1))

def test_sympy__physics__quantum__qft__RkGate():
    if False:
        while True:
            i = 10
    from sympy.physics.quantum.qft import RkGate
    assert _test_args(RkGate(0, 1))

def test_sympy__physics__quantum__qubit__IntQubit():
    if False:
        print('Hello World!')
    from sympy.physics.quantum.qubit import IntQubit
    assert _test_args(IntQubit(0))

def test_sympy__physics__quantum__qubit__IntQubitBra():
    if False:
        for i in range(10):
            print('nop')
    from sympy.physics.quantum.qubit import IntQubitBra
    assert _test_args(IntQubitBra(0))

def test_sympy__physics__quantum__qubit__IntQubitState():
    if False:
        i = 10
        return i + 15
    from sympy.physics.quantum.qubit import IntQubitState, QubitState
    assert _test_args(IntQubitState(QubitState(0, 1)))

def test_sympy__physics__quantum__qubit__Qubit():
    if False:
        i = 10
        return i + 15
    from sympy.physics.quantum.qubit import Qubit
    assert _test_args(Qubit(0, 0, 0))

def test_sympy__physics__quantum__qubit__QubitBra():
    if False:
        while True:
            i = 10
    from sympy.physics.quantum.qubit import QubitBra
    assert _test_args(QubitBra('1', 0))

def test_sympy__physics__quantum__qubit__QubitState():
    if False:
        return 10
    from sympy.physics.quantum.qubit import QubitState
    assert _test_args(QubitState(0, 1))

def test_sympy__physics__quantum__density__Density():
    if False:
        i = 10
        return i + 15
    from sympy.physics.quantum.density import Density
    from sympy.physics.quantum.state import Ket
    assert _test_args(Density([Ket(0), 0.5], [Ket(1), 0.5]))

@SKIP('TODO: sympy.physics.quantum.shor: Cmod Not Implemented')
def test_sympy__physics__quantum__shor__CMod():
    if False:
        for i in range(10):
            print('nop')
    from sympy.physics.quantum.shor import CMod
    assert _test_args(CMod())

def test_sympy__physics__quantum__spin__CoupledSpinState():
    if False:
        return 10
    from sympy.physics.quantum.spin import CoupledSpinState
    assert _test_args(CoupledSpinState(1, 0, (1, 1)))
    assert _test_args(CoupledSpinState(1, 0, (1, S.Half, S.Half)))
    assert _test_args(CoupledSpinState(1, 0, (1, S.Half, S.Half), ((2, 3, S.Half), (1, 2, 1))))
    (j, m, j1, j2, j3, j12, x) = symbols('j m j1:4 j12 x')
    assert CoupledSpinState(j, m, (j1, j2, j3)).subs(j2, x) == CoupledSpinState(j, m, (j1, x, j3))
    assert CoupledSpinState(j, m, (j1, j2, j3), ((1, 3, j12), (1, 2, j))).subs(j12, x) == CoupledSpinState(j, m, (j1, j2, j3), ((1, 3, x), (1, 2, j)))

def test_sympy__physics__quantum__spin__J2Op():
    if False:
        while True:
            i = 10
    from sympy.physics.quantum.spin import J2Op
    assert _test_args(J2Op('J'))

def test_sympy__physics__quantum__spin__JminusOp():
    if False:
        for i in range(10):
            print('nop')
    from sympy.physics.quantum.spin import JminusOp
    assert _test_args(JminusOp('J'))

def test_sympy__physics__quantum__spin__JplusOp():
    if False:
        for i in range(10):
            print('nop')
    from sympy.physics.quantum.spin import JplusOp
    assert _test_args(JplusOp('J'))

def test_sympy__physics__quantum__spin__JxBra():
    if False:
        while True:
            i = 10
    from sympy.physics.quantum.spin import JxBra
    assert _test_args(JxBra(1, 0))

def test_sympy__physics__quantum__spin__JxBraCoupled():
    if False:
        return 10
    from sympy.physics.quantum.spin import JxBraCoupled
    assert _test_args(JxBraCoupled(1, 0, (1, 1)))

def test_sympy__physics__quantum__spin__JxKet():
    if False:
        while True:
            i = 10
    from sympy.physics.quantum.spin import JxKet
    assert _test_args(JxKet(1, 0))

def test_sympy__physics__quantum__spin__JxKetCoupled():
    if False:
        i = 10
        return i + 15
    from sympy.physics.quantum.spin import JxKetCoupled
    assert _test_args(JxKetCoupled(1, 0, (1, 1)))

def test_sympy__physics__quantum__spin__JxOp():
    if False:
        while True:
            i = 10
    from sympy.physics.quantum.spin import JxOp
    assert _test_args(JxOp('J'))

def test_sympy__physics__quantum__spin__JyBra():
    if False:
        while True:
            i = 10
    from sympy.physics.quantum.spin import JyBra
    assert _test_args(JyBra(1, 0))

def test_sympy__physics__quantum__spin__JyBraCoupled():
    if False:
        while True:
            i = 10
    from sympy.physics.quantum.spin import JyBraCoupled
    assert _test_args(JyBraCoupled(1, 0, (1, 1)))

def test_sympy__physics__quantum__spin__JyKet():
    if False:
        print('Hello World!')
    from sympy.physics.quantum.spin import JyKet
    assert _test_args(JyKet(1, 0))

def test_sympy__physics__quantum__spin__JyKetCoupled():
    if False:
        while True:
            i = 10
    from sympy.physics.quantum.spin import JyKetCoupled
    assert _test_args(JyKetCoupled(1, 0, (1, 1)))

def test_sympy__physics__quantum__spin__JyOp():
    if False:
        return 10
    from sympy.physics.quantum.spin import JyOp
    assert _test_args(JyOp('J'))

def test_sympy__physics__quantum__spin__JzBra():
    if False:
        print('Hello World!')
    from sympy.physics.quantum.spin import JzBra
    assert _test_args(JzBra(1, 0))

def test_sympy__physics__quantum__spin__JzBraCoupled():
    if False:
        for i in range(10):
            print('nop')
    from sympy.physics.quantum.spin import JzBraCoupled
    assert _test_args(JzBraCoupled(1, 0, (1, 1)))

def test_sympy__physics__quantum__spin__JzKet():
    if False:
        for i in range(10):
            print('nop')
    from sympy.physics.quantum.spin import JzKet
    assert _test_args(JzKet(1, 0))

def test_sympy__physics__quantum__spin__JzKetCoupled():
    if False:
        while True:
            i = 10
    from sympy.physics.quantum.spin import JzKetCoupled
    assert _test_args(JzKetCoupled(1, 0, (1, 1)))

def test_sympy__physics__quantum__spin__JzOp():
    if False:
        print('Hello World!')
    from sympy.physics.quantum.spin import JzOp
    assert _test_args(JzOp('J'))

def test_sympy__physics__quantum__spin__Rotation():
    if False:
        print('Hello World!')
    from sympy.physics.quantum.spin import Rotation
    assert _test_args(Rotation(pi, 0, pi / 2))

def test_sympy__physics__quantum__spin__SpinState():
    if False:
        print('Hello World!')
    from sympy.physics.quantum.spin import SpinState
    assert _test_args(SpinState(1, 0))

def test_sympy__physics__quantum__spin__WignerD():
    if False:
        i = 10
        return i + 15
    from sympy.physics.quantum.spin import WignerD
    assert _test_args(WignerD(0, 1, 2, 3, 4, 5))

def test_sympy__physics__quantum__state__Bra():
    if False:
        return 10
    from sympy.physics.quantum.state import Bra
    assert _test_args(Bra(0))

def test_sympy__physics__quantum__state__BraBase():
    if False:
        while True:
            i = 10
    from sympy.physics.quantum.state import BraBase
    assert _test_args(BraBase(0))

def test_sympy__physics__quantum__state__Ket():
    if False:
        return 10
    from sympy.physics.quantum.state import Ket
    assert _test_args(Ket(0))

def test_sympy__physics__quantum__state__KetBase():
    if False:
        while True:
            i = 10
    from sympy.physics.quantum.state import KetBase
    assert _test_args(KetBase(0))

def test_sympy__physics__quantum__state__State():
    if False:
        i = 10
        return i + 15
    from sympy.physics.quantum.state import State
    assert _test_args(State(0))

def test_sympy__physics__quantum__state__StateBase():
    if False:
        while True:
            i = 10
    from sympy.physics.quantum.state import StateBase
    assert _test_args(StateBase(0))

def test_sympy__physics__quantum__state__OrthogonalBra():
    if False:
        while True:
            i = 10
    from sympy.physics.quantum.state import OrthogonalBra
    assert _test_args(OrthogonalBra(0))

def test_sympy__physics__quantum__state__OrthogonalKet():
    if False:
        i = 10
        return i + 15
    from sympy.physics.quantum.state import OrthogonalKet
    assert _test_args(OrthogonalKet(0))

def test_sympy__physics__quantum__state__OrthogonalState():
    if False:
        i = 10
        return i + 15
    from sympy.physics.quantum.state import OrthogonalState
    assert _test_args(OrthogonalState(0))

def test_sympy__physics__quantum__state__TimeDepBra():
    if False:
        for i in range(10):
            print('nop')
    from sympy.physics.quantum.state import TimeDepBra
    assert _test_args(TimeDepBra('psi', 't'))

def test_sympy__physics__quantum__state__TimeDepKet():
    if False:
        return 10
    from sympy.physics.quantum.state import TimeDepKet
    assert _test_args(TimeDepKet('psi', 't'))

def test_sympy__physics__quantum__state__TimeDepState():
    if False:
        for i in range(10):
            print('nop')
    from sympy.physics.quantum.state import TimeDepState
    assert _test_args(TimeDepState('psi', 't'))

def test_sympy__physics__quantum__state__Wavefunction():
    if False:
        return 10
    from sympy.physics.quantum.state import Wavefunction
    from sympy.functions import sin
    from sympy.functions.elementary.piecewise import Piecewise
    n = 1
    L = 1
    g = Piecewise((0, x < 0), (0, x > L), (sqrt(2 // L) * sin(n * pi * x / L), True))
    assert _test_args(Wavefunction(g, x))

def test_sympy__physics__quantum__tensorproduct__TensorProduct():
    if False:
        for i in range(10):
            print('nop')
    from sympy.physics.quantum.tensorproduct import TensorProduct
    (x, y) = symbols('x y', commutative=False)
    assert _test_args(TensorProduct(x, y))

def test_sympy__physics__quantum__identitysearch__GateIdentity():
    if False:
        return 10
    from sympy.physics.quantum.gate import X
    from sympy.physics.quantum.identitysearch import GateIdentity
    assert _test_args(GateIdentity(X(0), X(0)))

def test_sympy__physics__quantum__sho1d__SHOOp():
    if False:
        i = 10
        return i + 15
    from sympy.physics.quantum.sho1d import SHOOp
    assert _test_args(SHOOp('a'))

def test_sympy__physics__quantum__sho1d__RaisingOp():
    if False:
        for i in range(10):
            print('nop')
    from sympy.physics.quantum.sho1d import RaisingOp
    assert _test_args(RaisingOp('a'))

def test_sympy__physics__quantum__sho1d__LoweringOp():
    if False:
        while True:
            i = 10
    from sympy.physics.quantum.sho1d import LoweringOp
    assert _test_args(LoweringOp('a'))

def test_sympy__physics__quantum__sho1d__NumberOp():
    if False:
        for i in range(10):
            print('nop')
    from sympy.physics.quantum.sho1d import NumberOp
    assert _test_args(NumberOp('N'))

def test_sympy__physics__quantum__sho1d__Hamiltonian():
    if False:
        i = 10
        return i + 15
    from sympy.physics.quantum.sho1d import Hamiltonian
    assert _test_args(Hamiltonian('H'))

def test_sympy__physics__quantum__sho1d__SHOState():
    if False:
        i = 10
        return i + 15
    from sympy.physics.quantum.sho1d import SHOState
    assert _test_args(SHOState(0))

def test_sympy__physics__quantum__sho1d__SHOKet():
    if False:
        return 10
    from sympy.physics.quantum.sho1d import SHOKet
    assert _test_args(SHOKet(0))

def test_sympy__physics__quantum__sho1d__SHOBra():
    if False:
        i = 10
        return i + 15
    from sympy.physics.quantum.sho1d import SHOBra
    assert _test_args(SHOBra(0))

def test_sympy__physics__secondquant__AnnihilateBoson():
    if False:
        i = 10
        return i + 15
    from sympy.physics.secondquant import AnnihilateBoson
    assert _test_args(AnnihilateBoson(0))

def test_sympy__physics__secondquant__AnnihilateFermion():
    if False:
        while True:
            i = 10
    from sympy.physics.secondquant import AnnihilateFermion
    assert _test_args(AnnihilateFermion(0))

@SKIP('abstract class')
def test_sympy__physics__secondquant__Annihilator():
    if False:
        print('Hello World!')
    pass

def test_sympy__physics__secondquant__AntiSymmetricTensor():
    if False:
        while True:
            i = 10
    from sympy.physics.secondquant import AntiSymmetricTensor
    (i, j) = symbols('i j', below_fermi=True)
    (a, b) = symbols('a b', above_fermi=True)
    assert _test_args(AntiSymmetricTensor('v', (a, i), (b, j)))

def test_sympy__physics__secondquant__BosonState():
    if False:
        for i in range(10):
            print('nop')
    from sympy.physics.secondquant import BosonState
    assert _test_args(BosonState((0, 1)))

@SKIP('abstract class')
def test_sympy__physics__secondquant__BosonicOperator():
    if False:
        for i in range(10):
            print('nop')
    pass

def test_sympy__physics__secondquant__Commutator():
    if False:
        for i in range(10):
            print('nop')
    from sympy.physics.secondquant import Commutator
    (x, y) = symbols('x y', commutative=False)
    assert _test_args(Commutator(x, y))

def test_sympy__physics__secondquant__CreateBoson():
    if False:
        while True:
            i = 10
    from sympy.physics.secondquant import CreateBoson
    assert _test_args(CreateBoson(0))

def test_sympy__physics__secondquant__CreateFermion():
    if False:
        i = 10
        return i + 15
    from sympy.physics.secondquant import CreateFermion
    assert _test_args(CreateFermion(0))

@SKIP('abstract class')
def test_sympy__physics__secondquant__Creator():
    if False:
        while True:
            i = 10
    pass

def test_sympy__physics__secondquant__Dagger():
    if False:
        return 10
    from sympy.physics.secondquant import Dagger
    assert _test_args(Dagger(x))

def test_sympy__physics__secondquant__FermionState():
    if False:
        return 10
    from sympy.physics.secondquant import FermionState
    assert _test_args(FermionState((0, 1)))

def test_sympy__physics__secondquant__FermionicOperator():
    if False:
        i = 10
        return i + 15
    from sympy.physics.secondquant import FermionicOperator
    assert _test_args(FermionicOperator(0))

def test_sympy__physics__secondquant__FockState():
    if False:
        for i in range(10):
            print('nop')
    from sympy.physics.secondquant import FockState
    assert _test_args(FockState((0, 1)))

def test_sympy__physics__secondquant__FockStateBosonBra():
    if False:
        while True:
            i = 10
    from sympy.physics.secondquant import FockStateBosonBra
    assert _test_args(FockStateBosonBra((0, 1)))

def test_sympy__physics__secondquant__FockStateBosonKet():
    if False:
        i = 10
        return i + 15
    from sympy.physics.secondquant import FockStateBosonKet
    assert _test_args(FockStateBosonKet((0, 1)))

def test_sympy__physics__secondquant__FockStateBra():
    if False:
        for i in range(10):
            print('nop')
    from sympy.physics.secondquant import FockStateBra
    assert _test_args(FockStateBra((0, 1)))

def test_sympy__physics__secondquant__FockStateFermionBra():
    if False:
        i = 10
        return i + 15
    from sympy.physics.secondquant import FockStateFermionBra
    assert _test_args(FockStateFermionBra((0, 1)))

def test_sympy__physics__secondquant__FockStateFermionKet():
    if False:
        while True:
            i = 10
    from sympy.physics.secondquant import FockStateFermionKet
    assert _test_args(FockStateFermionKet((0, 1)))

def test_sympy__physics__secondquant__FockStateKet():
    if False:
        i = 10
        return i + 15
    from sympy.physics.secondquant import FockStateKet
    assert _test_args(FockStateKet((0, 1)))

def test_sympy__physics__secondquant__InnerProduct():
    if False:
        i = 10
        return i + 15
    from sympy.physics.secondquant import InnerProduct
    from sympy.physics.secondquant import FockStateKet, FockStateBra
    assert _test_args(InnerProduct(FockStateBra((0, 1)), FockStateKet((0, 1))))

def test_sympy__physics__secondquant__NO():
    if False:
        print('Hello World!')
    from sympy.physics.secondquant import NO, F, Fd
    assert _test_args(NO(Fd(x) * F(y)))

def test_sympy__physics__secondquant__PermutationOperator():
    if False:
        print('Hello World!')
    from sympy.physics.secondquant import PermutationOperator
    assert _test_args(PermutationOperator(0, 1))

def test_sympy__physics__secondquant__SqOperator():
    if False:
        while True:
            i = 10
    from sympy.physics.secondquant import SqOperator
    assert _test_args(SqOperator(0))

def test_sympy__physics__secondquant__TensorSymbol():
    if False:
        for i in range(10):
            print('nop')
    from sympy.physics.secondquant import TensorSymbol
    assert _test_args(TensorSymbol(x))

def test_sympy__physics__control__lti__LinearTimeInvariant():
    if False:
        while True:
            i = 10
    pass

def test_sympy__physics__control__lti__SISOLinearTimeInvariant():
    if False:
        i = 10
        return i + 15
    pass

def test_sympy__physics__control__lti__MIMOLinearTimeInvariant():
    if False:
        while True:
            i = 10
    pass

def test_sympy__physics__control__lti__TransferFunction():
    if False:
        for i in range(10):
            print('nop')
    from sympy.physics.control.lti import TransferFunction
    assert _test_args(TransferFunction(2, 3, x))

def test_sympy__physics__control__lti__Series():
    if False:
        while True:
            i = 10
    from sympy.physics.control import Series, TransferFunction
    tf1 = TransferFunction(x ** 2 - y ** 3, y - z, x)
    tf2 = TransferFunction(y - x, z + y, x)
    assert _test_args(Series(tf1, tf2))

def test_sympy__physics__control__lti__MIMOSeries():
    if False:
        return 10
    from sympy.physics.control import MIMOSeries, TransferFunction, TransferFunctionMatrix
    tf1 = TransferFunction(x ** 2 - y ** 3, y - z, x)
    tf2 = TransferFunction(y - x, z + y, x)
    tfm_1 = TransferFunctionMatrix([[tf2, tf1]])
    tfm_2 = TransferFunctionMatrix([[tf1, tf2], [tf2, tf1]])
    tfm_3 = TransferFunctionMatrix([[tf1], [tf2]])
    assert _test_args(MIMOSeries(tfm_3, tfm_2, tfm_1))

def test_sympy__physics__control__lti__Parallel():
    if False:
        return 10
    from sympy.physics.control import Parallel, TransferFunction
    tf1 = TransferFunction(x ** 2 - y ** 3, y - z, x)
    tf2 = TransferFunction(y - x, z + y, x)
    assert _test_args(Parallel(tf1, tf2))

def test_sympy__physics__control__lti__MIMOParallel():
    if False:
        for i in range(10):
            print('nop')
    from sympy.physics.control import MIMOParallel, TransferFunction, TransferFunctionMatrix
    tf1 = TransferFunction(x ** 2 - y ** 3, y - z, x)
    tf2 = TransferFunction(y - x, z + y, x)
    tfm_1 = TransferFunctionMatrix([[tf1, tf2], [tf2, tf1]])
    tfm_2 = TransferFunctionMatrix([[tf2, tf1], [tf1, tf2]])
    assert _test_args(MIMOParallel(tfm_1, tfm_2))

def test_sympy__physics__control__lti__Feedback():
    if False:
        for i in range(10):
            print('nop')
    from sympy.physics.control import TransferFunction, Feedback
    tf1 = TransferFunction(x ** 2 - y ** 3, y - z, x)
    tf2 = TransferFunction(y - x, z + y, x)
    assert _test_args(Feedback(tf1, tf2))
    assert _test_args(Feedback(tf1, tf2, 1))

def test_sympy__physics__control__lti__MIMOFeedback():
    if False:
        i = 10
        return i + 15
    from sympy.physics.control import TransferFunction, MIMOFeedback, TransferFunctionMatrix
    tf1 = TransferFunction(x ** 2 - y ** 3, y - z, x)
    tf2 = TransferFunction(y - x, z + y, x)
    tfm_1 = TransferFunctionMatrix([[tf2, tf1], [tf1, tf2]])
    tfm_2 = TransferFunctionMatrix([[tf1, tf2], [tf2, tf1]])
    assert _test_args(MIMOFeedback(tfm_1, tfm_2))
    assert _test_args(MIMOFeedback(tfm_1, tfm_2, 1))

def test_sympy__physics__control__lti__TransferFunctionMatrix():
    if False:
        i = 10
        return i + 15
    from sympy.physics.control import TransferFunction, TransferFunctionMatrix
    tf1 = TransferFunction(x ** 2 - y ** 3, y - z, x)
    tf2 = TransferFunction(y - x, z + y, x)
    assert _test_args(TransferFunctionMatrix([[tf1, tf2]]))

def test_sympy__physics__control__lti__StateSpace():
    if False:
        for i in range(10):
            print('nop')
    from sympy.matrices.dense import Matrix
    from sympy.physics.control import StateSpace
    A = Matrix([[-5, -1], [3, -1]])
    B = Matrix([2, 5])
    C = Matrix([[1, 2]])
    D = Matrix([0])
    assert _test_args(StateSpace(A, B, C, D))

def test_sympy__physics__units__dimensions__Dimension():
    if False:
        while True:
            i = 10
    from sympy.physics.units.dimensions import Dimension
    assert _test_args(Dimension('length', 'L'))

def test_sympy__physics__units__dimensions__DimensionSystem():
    if False:
        while True:
            i = 10
    from sympy.physics.units.dimensions import DimensionSystem
    from sympy.physics.units.definitions.dimension_definitions import length, time, velocity
    assert _test_args(DimensionSystem((length, time), (velocity,)))

def test_sympy__physics__units__quantities__Quantity():
    if False:
        for i in range(10):
            print('nop')
    from sympy.physics.units.quantities import Quantity
    assert _test_args(Quantity('dam'))

def test_sympy__physics__units__quantities__PhysicalConstant():
    if False:
        while True:
            i = 10
    from sympy.physics.units.quantities import PhysicalConstant
    assert _test_args(PhysicalConstant('foo'))

def test_sympy__physics__units__prefixes__Prefix():
    if False:
        return 10
    from sympy.physics.units.prefixes import Prefix
    assert _test_args(Prefix('kilo', 'k', 3))

def test_sympy__core__numbers__AlgebraicNumber():
    if False:
        i = 10
        return i + 15
    from sympy.core.numbers import AlgebraicNumber
    assert _test_args(AlgebraicNumber(sqrt(2), [1, 2, 3]))

def test_sympy__polys__polytools__GroebnerBasis():
    if False:
        for i in range(10):
            print('nop')
    from sympy.polys.polytools import GroebnerBasis
    assert _test_args(GroebnerBasis([x, y, z], x, y, z))

def test_sympy__polys__polytools__Poly():
    if False:
        i = 10
        return i + 15
    from sympy.polys.polytools import Poly
    assert _test_args(Poly(2, x, y))

def test_sympy__polys__polytools__PurePoly():
    if False:
        i = 10
        return i + 15
    from sympy.polys.polytools import PurePoly
    assert _test_args(PurePoly(2, x, y))

@SKIP('abstract class')
def test_sympy__polys__rootoftools__RootOf():
    if False:
        print('Hello World!')
    pass

def test_sympy__polys__rootoftools__ComplexRootOf():
    if False:
        print('Hello World!')
    from sympy.polys.rootoftools import ComplexRootOf
    assert _test_args(ComplexRootOf(x ** 3 + x + 1, 0))

def test_sympy__polys__rootoftools__RootSum():
    if False:
        i = 10
        return i + 15
    from sympy.polys.rootoftools import RootSum
    assert _test_args(RootSum(x ** 3 + x + 1, sin))

def test_sympy__series__limits__Limit():
    if False:
        return 10
    from sympy.series.limits import Limit
    assert _test_args(Limit(x, x, 0, dir='-'))

def test_sympy__series__order__Order():
    if False:
        for i in range(10):
            print('nop')
    from sympy.series.order import Order
    assert _test_args(Order(1, x, y))

@SKIP('Abstract Class')
def test_sympy__series__sequences__SeqBase():
    if False:
        return 10
    pass

def test_sympy__series__sequences__EmptySequence():
    if False:
        return 10
    from sympy.series import EmptySequence
    assert _test_args(EmptySequence)

@SKIP('Abstract Class')
def test_sympy__series__sequences__SeqExpr():
    if False:
        print('Hello World!')
    pass

def test_sympy__series__sequences__SeqPer():
    if False:
        return 10
    from sympy.series.sequences import SeqPer
    assert _test_args(SeqPer((1, 2, 3), (0, 10)))

def test_sympy__series__sequences__SeqFormula():
    if False:
        for i in range(10):
            print('nop')
    from sympy.series.sequences import SeqFormula
    assert _test_args(SeqFormula(x ** 2, (0, 10)))

def test_sympy__series__sequences__RecursiveSeq():
    if False:
        return 10
    from sympy.series.sequences import RecursiveSeq
    y = Function('y')
    n = symbols('n')
    assert _test_args(RecursiveSeq(y(n - 1) + y(n - 2), y(n), n, (0, 1)))
    assert _test_args(RecursiveSeq(y(n - 1) + y(n - 2), y(n), n))

def test_sympy__series__sequences__SeqExprOp():
    if False:
        return 10
    from sympy.series.sequences import SeqExprOp, sequence
    s1 = sequence((1, 2, 3))
    s2 = sequence(x ** 2)
    assert _test_args(SeqExprOp(s1, s2))

def test_sympy__series__sequences__SeqAdd():
    if False:
        while True:
            i = 10
    from sympy.series.sequences import SeqAdd, sequence
    s1 = sequence((1, 2, 3))
    s2 = sequence(x ** 2)
    assert _test_args(SeqAdd(s1, s2))

def test_sympy__series__sequences__SeqMul():
    if False:
        i = 10
        return i + 15
    from sympy.series.sequences import SeqMul, sequence
    s1 = sequence((1, 2, 3))
    s2 = sequence(x ** 2)
    assert _test_args(SeqMul(s1, s2))

@SKIP('Abstract Class')
def test_sympy__series__series_class__SeriesBase():
    if False:
        for i in range(10):
            print('nop')
    pass

def test_sympy__series__fourier__FourierSeries():
    if False:
        while True:
            i = 10
    from sympy.series.fourier import fourier_series
    assert _test_args(fourier_series(x, (x, -pi, pi)))

def test_sympy__series__fourier__FiniteFourierSeries():
    if False:
        for i in range(10):
            print('nop')
    from sympy.series.fourier import fourier_series
    assert _test_args(fourier_series(sin(pi * x), (x, -1, 1)))

def test_sympy__series__formal__FormalPowerSeries():
    if False:
        print('Hello World!')
    from sympy.series.formal import fps
    assert _test_args(fps(log(1 + x), x))

def test_sympy__series__formal__Coeff():
    if False:
        while True:
            i = 10
    from sympy.series.formal import fps
    assert _test_args(fps(x ** 2 + x + 1, x))

@SKIP('Abstract Class')
def test_sympy__series__formal__FiniteFormalPowerSeries():
    if False:
        return 10
    pass

def test_sympy__series__formal__FormalPowerSeriesProduct():
    if False:
        for i in range(10):
            print('nop')
    from sympy.series.formal import fps
    (f1, f2) = (fps(sin(x)), fps(exp(x)))
    assert _test_args(f1.product(f2, x))

def test_sympy__series__formal__FormalPowerSeriesCompose():
    if False:
        print('Hello World!')
    from sympy.series.formal import fps
    (f1, f2) = (fps(exp(x)), fps(sin(x)))
    assert _test_args(f1.compose(f2, x))

def test_sympy__series__formal__FormalPowerSeriesInverse():
    if False:
        print('Hello World!')
    from sympy.series.formal import fps
    f1 = fps(exp(x))
    assert _test_args(f1.inverse(x))

def test_sympy__simplify__hyperexpand__Hyper_Function():
    if False:
        print('Hello World!')
    from sympy.simplify.hyperexpand import Hyper_Function
    assert _test_args(Hyper_Function([2], [1]))

def test_sympy__simplify__hyperexpand__G_Function():
    if False:
        i = 10
        return i + 15
    from sympy.simplify.hyperexpand import G_Function
    assert _test_args(G_Function([2], [1], [], []))

@SKIP('abstract class')
def test_sympy__tensor__array__ndim_array__ImmutableNDimArray():
    if False:
        return 10
    pass

def test_sympy__tensor__array__dense_ndim_array__ImmutableDenseNDimArray():
    if False:
        i = 10
        return i + 15
    from sympy.tensor.array.dense_ndim_array import ImmutableDenseNDimArray
    densarr = ImmutableDenseNDimArray(range(10, 34), (2, 3, 4))
    assert _test_args(densarr)

def test_sympy__tensor__array__sparse_ndim_array__ImmutableSparseNDimArray():
    if False:
        i = 10
        return i + 15
    from sympy.tensor.array.sparse_ndim_array import ImmutableSparseNDimArray
    sparr = ImmutableSparseNDimArray(range(10, 34), (2, 3, 4))
    assert _test_args(sparr)

def test_sympy__tensor__array__array_comprehension__ArrayComprehension():
    if False:
        for i in range(10):
            print('nop')
    from sympy.tensor.array.array_comprehension import ArrayComprehension
    arrcom = ArrayComprehension(x, (x, 1, 5))
    assert _test_args(arrcom)

def test_sympy__tensor__array__array_comprehension__ArrayComprehensionMap():
    if False:
        return 10
    from sympy.tensor.array.array_comprehension import ArrayComprehensionMap
    arrcomma = ArrayComprehensionMap(lambda : 0, (x, 1, 5))
    assert _test_args(arrcomma)

def test_sympy__tensor__array__array_derivatives__ArrayDerivative():
    if False:
        for i in range(10):
            print('nop')
    from sympy.tensor.array.array_derivatives import ArrayDerivative
    A = MatrixSymbol('A', 2, 2)
    arrder = ArrayDerivative(A, A, evaluate=False)
    assert _test_args(arrder)

def test_sympy__tensor__array__expressions__array_expressions__ArraySymbol():
    if False:
        while True:
            i = 10
    from sympy.tensor.array.expressions.array_expressions import ArraySymbol
    (m, n, k) = symbols('m n k')
    array = ArraySymbol('A', (m, n, k, 2))
    assert _test_args(array)

def test_sympy__tensor__array__expressions__array_expressions__ArrayElement():
    if False:
        return 10
    from sympy.tensor.array.expressions.array_expressions import ArrayElement
    (m, n, k) = symbols('m n k')
    ae = ArrayElement('A', (m, n, k, 2))
    assert _test_args(ae)

def test_sympy__tensor__array__expressions__array_expressions__ZeroArray():
    if False:
        return 10
    from sympy.tensor.array.expressions.array_expressions import ZeroArray
    (m, n, k) = symbols('m n k')
    za = ZeroArray(m, n, k, 2)
    assert _test_args(za)

def test_sympy__tensor__array__expressions__array_expressions__OneArray():
    if False:
        return 10
    from sympy.tensor.array.expressions.array_expressions import OneArray
    (m, n, k) = symbols('m n k')
    za = OneArray(m, n, k, 2)
    assert _test_args(za)

def test_sympy__tensor__functions__TensorProduct():
    if False:
        print('Hello World!')
    from sympy.tensor.functions import TensorProduct
    A = MatrixSymbol('A', 3, 3)
    B = MatrixSymbol('B', 3, 3)
    tp = TensorProduct(A, B)
    assert _test_args(tp)

def test_sympy__tensor__indexed__Idx():
    if False:
        print('Hello World!')
    from sympy.tensor.indexed import Idx
    assert _test_args(Idx('test'))
    assert _test_args(Idx('test', (0, 10)))
    assert _test_args(Idx('test', 2))
    assert _test_args(Idx('test', x))

def test_sympy__tensor__indexed__Indexed():
    if False:
        return 10
    from sympy.tensor.indexed import Indexed, Idx
    assert _test_args(Indexed('A', Idx('i'), Idx('j')))

def test_sympy__tensor__indexed__IndexedBase():
    if False:
        for i in range(10):
            print('nop')
    from sympy.tensor.indexed import IndexedBase
    assert _test_args(IndexedBase('A', shape=(x, y)))
    assert _test_args(IndexedBase('A', 1))
    assert _test_args(IndexedBase('A')[0, 1])

def test_sympy__tensor__tensor__TensorIndexType():
    if False:
        print('Hello World!')
    from sympy.tensor.tensor import TensorIndexType
    assert _test_args(TensorIndexType('Lorentz'))

@SKIP('deprecated class')
def test_sympy__tensor__tensor__TensorType():
    if False:
        return 10
    pass

def test_sympy__tensor__tensor__TensorSymmetry():
    if False:
        i = 10
        return i + 15
    from sympy.tensor.tensor import TensorSymmetry, get_symmetric_group_sgs
    assert _test_args(TensorSymmetry(get_symmetric_group_sgs(2)))

def test_sympy__tensor__tensor__TensorHead():
    if False:
        while True:
            i = 10
    from sympy.tensor.tensor import TensorIndexType, TensorSymmetry, get_symmetric_group_sgs, TensorHead
    Lorentz = TensorIndexType('Lorentz', dummy_name='L')
    sym = TensorSymmetry(get_symmetric_group_sgs(1))
    assert _test_args(TensorHead('p', [Lorentz], sym, 0))

def test_sympy__tensor__tensor__TensorIndex():
    if False:
        for i in range(10):
            print('nop')
    from sympy.tensor.tensor import TensorIndexType, TensorIndex
    Lorentz = TensorIndexType('Lorentz', dummy_name='L')
    assert _test_args(TensorIndex('i', Lorentz))

@SKIP('abstract class')
def test_sympy__tensor__tensor__TensExpr():
    if False:
        while True:
            i = 10
    pass

def test_sympy__tensor__tensor__TensAdd():
    if False:
        return 10
    from sympy.tensor.tensor import TensorIndexType, TensorSymmetry, get_symmetric_group_sgs, tensor_indices, TensAdd, tensor_heads
    Lorentz = TensorIndexType('Lorentz', dummy_name='L')
    (a, b) = tensor_indices('a,b', Lorentz)
    sym = TensorSymmetry(get_symmetric_group_sgs(1))
    (p, q) = tensor_heads('p,q', [Lorentz], sym)
    t1 = p(a)
    t2 = q(a)
    assert _test_args(TensAdd(t1, t2))

def test_sympy__tensor__tensor__Tensor():
    if False:
        return 10
    from sympy.tensor.tensor import TensorIndexType, TensorSymmetry, get_symmetric_group_sgs, tensor_indices, TensorHead
    Lorentz = TensorIndexType('Lorentz', dummy_name='L')
    (a, b) = tensor_indices('a,b', Lorentz)
    sym = TensorSymmetry(get_symmetric_group_sgs(1))
    p = TensorHead('p', [Lorentz], sym)
    assert _test_args(p(a))

def test_sympy__tensor__tensor__TensMul():
    if False:
        for i in range(10):
            print('nop')
    from sympy.tensor.tensor import TensorIndexType, TensorSymmetry, get_symmetric_group_sgs, tensor_indices, tensor_heads
    Lorentz = TensorIndexType('Lorentz', dummy_name='L')
    (a, b) = tensor_indices('a,b', Lorentz)
    sym = TensorSymmetry(get_symmetric_group_sgs(1))
    (p, q) = tensor_heads('p, q', [Lorentz], sym)
    assert _test_args(3 * p(a) * q(b))

def test_sympy__tensor__tensor__TensorElement():
    if False:
        for i in range(10):
            print('nop')
    from sympy.tensor.tensor import TensorIndexType, TensorHead, TensorElement
    L = TensorIndexType('L')
    A = TensorHead('A', [L, L])
    telem = TensorElement(A(x, y), {x: 1})
    assert _test_args(telem)

def test_sympy__tensor__tensor__WildTensor():
    if False:
        i = 10
        return i + 15
    from sympy.tensor.tensor import TensorIndexType, WildTensorHead, TensorIndex
    Lorentz = TensorIndexType('Lorentz', dummy_name='L')
    a = TensorIndex('a', Lorentz)
    p = WildTensorHead('p')
    assert _test_args(p(a))

def test_sympy__tensor__tensor__WildTensorHead():
    if False:
        i = 10
        return i + 15
    from sympy.tensor.tensor import WildTensorHead
    assert _test_args(WildTensorHead('p'))

def test_sympy__tensor__tensor__WildTensorIndex():
    if False:
        return 10
    from sympy.tensor.tensor import TensorIndexType, WildTensorIndex
    Lorentz = TensorIndexType('Lorentz', dummy_name='L')
    assert _test_args(WildTensorIndex('i', Lorentz))

def test_sympy__tensor__toperators__PartialDerivative():
    if False:
        for i in range(10):
            print('nop')
    from sympy.tensor.tensor import TensorIndexType, tensor_indices, TensorHead
    from sympy.tensor.toperators import PartialDerivative
    Lorentz = TensorIndexType('Lorentz', dummy_name='L')
    (a, b) = tensor_indices('a,b', Lorentz)
    A = TensorHead('A', [Lorentz])
    assert _test_args(PartialDerivative(A(a), A(b)))

def test_as_coeff_add():
    if False:
        while True:
            i = 10
    assert (7, (3 * x, 4 * x ** 2)) == (7 + 3 * x + 4 * x ** 2).as_coeff_add()

def test_sympy__geometry__curve__Curve():
    if False:
        return 10
    from sympy.geometry.curve import Curve
    assert _test_args(Curve((x, 1), (x, 0, 1)))

def test_sympy__geometry__point__Point():
    if False:
        i = 10
        return i + 15
    from sympy.geometry.point import Point
    assert _test_args(Point(0, 1))

def test_sympy__geometry__point__Point2D():
    if False:
        return 10
    from sympy.geometry.point import Point2D
    assert _test_args(Point2D(0, 1))

def test_sympy__geometry__point__Point3D():
    if False:
        print('Hello World!')
    from sympy.geometry.point import Point3D
    assert _test_args(Point3D(0, 1, 2))

def test_sympy__geometry__ellipse__Ellipse():
    if False:
        return 10
    from sympy.geometry.ellipse import Ellipse
    assert _test_args(Ellipse((0, 1), 2, 3))

def test_sympy__geometry__ellipse__Circle():
    if False:
        print('Hello World!')
    from sympy.geometry.ellipse import Circle
    assert _test_args(Circle((0, 1), 2))

def test_sympy__geometry__parabola__Parabola():
    if False:
        i = 10
        return i + 15
    from sympy.geometry.parabola import Parabola
    from sympy.geometry.line import Line
    assert _test_args(Parabola((0, 0), Line((2, 3), (4, 3))))

@SKIP('abstract class')
def test_sympy__geometry__line__LinearEntity():
    if False:
        i = 10
        return i + 15
    pass

def test_sympy__geometry__line__Line():
    if False:
        i = 10
        return i + 15
    from sympy.geometry.line import Line
    assert _test_args(Line((0, 1), (2, 3)))

def test_sympy__geometry__line__Ray():
    if False:
        return 10
    from sympy.geometry.line import Ray
    assert _test_args(Ray((0, 1), (2, 3)))

def test_sympy__geometry__line__Segment():
    if False:
        for i in range(10):
            print('nop')
    from sympy.geometry.line import Segment
    assert _test_args(Segment((0, 1), (2, 3)))

@SKIP('abstract class')
def test_sympy__geometry__line__LinearEntity2D():
    if False:
        print('Hello World!')
    pass

def test_sympy__geometry__line__Line2D():
    if False:
        for i in range(10):
            print('nop')
    from sympy.geometry.line import Line2D
    assert _test_args(Line2D((0, 1), (2, 3)))

def test_sympy__geometry__line__Ray2D():
    if False:
        for i in range(10):
            print('nop')
    from sympy.geometry.line import Ray2D
    assert _test_args(Ray2D((0, 1), (2, 3)))

def test_sympy__geometry__line__Segment2D():
    if False:
        return 10
    from sympy.geometry.line import Segment2D
    assert _test_args(Segment2D((0, 1), (2, 3)))

@SKIP('abstract class')
def test_sympy__geometry__line__LinearEntity3D():
    if False:
        for i in range(10):
            print('nop')
    pass

def test_sympy__geometry__line__Line3D():
    if False:
        i = 10
        return i + 15
    from sympy.geometry.line import Line3D
    assert _test_args(Line3D((0, 1, 1), (2, 3, 4)))

def test_sympy__geometry__line__Segment3D():
    if False:
        print('Hello World!')
    from sympy.geometry.line import Segment3D
    assert _test_args(Segment3D((0, 1, 1), (2, 3, 4)))

def test_sympy__geometry__line__Ray3D():
    if False:
        return 10
    from sympy.geometry.line import Ray3D
    assert _test_args(Ray3D((0, 1, 1), (2, 3, 4)))

def test_sympy__geometry__plane__Plane():
    if False:
        i = 10
        return i + 15
    from sympy.geometry.plane import Plane
    assert _test_args(Plane((1, 1, 1), (-3, 4, -2), (1, 2, 3)))

def test_sympy__geometry__polygon__Polygon():
    if False:
        i = 10
        return i + 15
    from sympy.geometry.polygon import Polygon
    assert _test_args(Polygon((0, 1), (2, 3), (4, 5), (6, 7)))

def test_sympy__geometry__polygon__RegularPolygon():
    if False:
        while True:
            i = 10
    from sympy.geometry.polygon import RegularPolygon
    assert _test_args(RegularPolygon((0, 1), 2, 3, 4))

def test_sympy__geometry__polygon__Triangle():
    if False:
        while True:
            i = 10
    from sympy.geometry.polygon import Triangle
    assert _test_args(Triangle((0, 1), (2, 3), (4, 5)))

def test_sympy__geometry__entity__GeometryEntity():
    if False:
        for i in range(10):
            print('nop')
    from sympy.geometry.entity import GeometryEntity
    from sympy.geometry.point import Point
    assert _test_args(GeometryEntity(Point(1, 0), 1, [1, 2]))

@SKIP('abstract class')
def test_sympy__geometry__entity__GeometrySet():
    if False:
        print('Hello World!')
    pass

def test_sympy__diffgeom__diffgeom__Manifold():
    if False:
        for i in range(10):
            print('nop')
    from sympy.diffgeom import Manifold
    assert _test_args(Manifold('name', 3))

def test_sympy__diffgeom__diffgeom__Patch():
    if False:
        print('Hello World!')
    from sympy.diffgeom import Manifold, Patch
    assert _test_args(Patch('name', Manifold('name', 3)))

def test_sympy__diffgeom__diffgeom__CoordSystem():
    if False:
        while True:
            i = 10
    from sympy.diffgeom import Manifold, Patch, CoordSystem
    assert _test_args(CoordSystem('name', Patch('name', Manifold('name', 3))))
    assert _test_args(CoordSystem('name', Patch('name', Manifold('name', 3)), [a, b, c]))

def test_sympy__diffgeom__diffgeom__CoordinateSymbol():
    if False:
        return 10
    from sympy.diffgeom import Manifold, Patch, CoordSystem, CoordinateSymbol
    assert _test_args(CoordinateSymbol(CoordSystem('name', Patch('name', Manifold('name', 3)), [a, b, c]), 0))

def test_sympy__diffgeom__diffgeom__Point():
    if False:
        i = 10
        return i + 15
    from sympy.diffgeom import Manifold, Patch, CoordSystem, Point
    assert _test_args(Point(CoordSystem('name', Patch('name', Manifold('name', 3)), [a, b, c]), [x, y]))

def test_sympy__diffgeom__diffgeom__BaseScalarField():
    if False:
        i = 10
        return i + 15
    from sympy.diffgeom import Manifold, Patch, CoordSystem, BaseScalarField
    cs = CoordSystem('name', Patch('name', Manifold('name', 3)), [a, b, c])
    assert _test_args(BaseScalarField(cs, 0))

def test_sympy__diffgeom__diffgeom__BaseVectorField():
    if False:
        return 10
    from sympy.diffgeom import Manifold, Patch, CoordSystem, BaseVectorField
    cs = CoordSystem('name', Patch('name', Manifold('name', 3)), [a, b, c])
    assert _test_args(BaseVectorField(cs, 0))

def test_sympy__diffgeom__diffgeom__Differential():
    if False:
        for i in range(10):
            print('nop')
    from sympy.diffgeom import Manifold, Patch, CoordSystem, BaseScalarField, Differential
    cs = CoordSystem('name', Patch('name', Manifold('name', 3)), [a, b, c])
    assert _test_args(Differential(BaseScalarField(cs, 0)))

def test_sympy__diffgeom__diffgeom__Commutator():
    if False:
        return 10
    from sympy.diffgeom import Manifold, Patch, CoordSystem, BaseVectorField, Commutator
    cs = CoordSystem('name', Patch('name', Manifold('name', 3)), [a, b, c])
    cs1 = CoordSystem('name1', Patch('name', Manifold('name', 3)), [a, b, c])
    v = BaseVectorField(cs, 0)
    v1 = BaseVectorField(cs1, 0)
    assert _test_args(Commutator(v, v1))

def test_sympy__diffgeom__diffgeom__TensorProduct():
    if False:
        print('Hello World!')
    from sympy.diffgeom import Manifold, Patch, CoordSystem, BaseScalarField, Differential, TensorProduct
    cs = CoordSystem('name', Patch('name', Manifold('name', 3)), [a, b, c])
    d = Differential(BaseScalarField(cs, 0))
    assert _test_args(TensorProduct(d, d))

def test_sympy__diffgeom__diffgeom__WedgeProduct():
    if False:
        i = 10
        return i + 15
    from sympy.diffgeom import Manifold, Patch, CoordSystem, BaseScalarField, Differential, WedgeProduct
    cs = CoordSystem('name', Patch('name', Manifold('name', 3)), [a, b, c])
    d = Differential(BaseScalarField(cs, 0))
    d1 = Differential(BaseScalarField(cs, 1))
    assert _test_args(WedgeProduct(d, d1))

def test_sympy__diffgeom__diffgeom__LieDerivative():
    if False:
        for i in range(10):
            print('nop')
    from sympy.diffgeom import Manifold, Patch, CoordSystem, BaseScalarField, Differential, BaseVectorField, LieDerivative
    cs = CoordSystem('name', Patch('name', Manifold('name', 3)), [a, b, c])
    d = Differential(BaseScalarField(cs, 0))
    v = BaseVectorField(cs, 0)
    assert _test_args(LieDerivative(v, d))

def test_sympy__diffgeom__diffgeom__BaseCovarDerivativeOp():
    if False:
        for i in range(10):
            print('nop')
    from sympy.diffgeom import Manifold, Patch, CoordSystem, BaseCovarDerivativeOp
    cs = CoordSystem('name', Patch('name', Manifold('name', 3)), [a, b, c])
    assert _test_args(BaseCovarDerivativeOp(cs, 0, [[[0] * 3] * 3] * 3))

def test_sympy__diffgeom__diffgeom__CovarDerivativeOp():
    if False:
        while True:
            i = 10
    from sympy.diffgeom import Manifold, Patch, CoordSystem, BaseVectorField, CovarDerivativeOp
    cs = CoordSystem('name', Patch('name', Manifold('name', 3)), [a, b, c])
    v = BaseVectorField(cs, 0)
    _test_args(CovarDerivativeOp(v, [[[0] * 3] * 3] * 3))

def test_sympy__categories__baseclasses__Class():
    if False:
        while True:
            i = 10
    from sympy.categories.baseclasses import Class
    assert _test_args(Class())

def test_sympy__categories__baseclasses__Object():
    if False:
        print('Hello World!')
    from sympy.categories import Object
    assert _test_args(Object('A'))

@SKIP('abstract class')
def test_sympy__categories__baseclasses__Morphism():
    if False:
        print('Hello World!')
    pass

def test_sympy__categories__baseclasses__IdentityMorphism():
    if False:
        print('Hello World!')
    from sympy.categories import Object, IdentityMorphism
    assert _test_args(IdentityMorphism(Object('A')))

def test_sympy__categories__baseclasses__NamedMorphism():
    if False:
        while True:
            i = 10
    from sympy.categories import Object, NamedMorphism
    assert _test_args(NamedMorphism(Object('A'), Object('B'), 'f'))

def test_sympy__categories__baseclasses__CompositeMorphism():
    if False:
        while True:
            i = 10
    from sympy.categories import Object, NamedMorphism, CompositeMorphism
    A = Object('A')
    B = Object('B')
    C = Object('C')
    f = NamedMorphism(A, B, 'f')
    g = NamedMorphism(B, C, 'g')
    assert _test_args(CompositeMorphism(f, g))

def test_sympy__categories__baseclasses__Diagram():
    if False:
        print('Hello World!')
    from sympy.categories import Object, NamedMorphism, Diagram
    A = Object('A')
    B = Object('B')
    f = NamedMorphism(A, B, 'f')
    d = Diagram([f])
    assert _test_args(d)

def test_sympy__categories__baseclasses__Category():
    if False:
        i = 10
        return i + 15
    from sympy.categories import Object, NamedMorphism, Diagram, Category
    A = Object('A')
    B = Object('B')
    C = Object('C')
    f = NamedMorphism(A, B, 'f')
    g = NamedMorphism(B, C, 'g')
    d1 = Diagram([f, g])
    d2 = Diagram([f])
    K = Category('K', commutative_diagrams=[d1, d2])
    assert _test_args(K)

def test_sympy__ntheory__factor___totient():
    if False:
        while True:
            i = 10
    from sympy.ntheory.factor_ import totient
    k = symbols('k', integer=True)
    t = totient(k)
    assert _test_args(t)

def test_sympy__ntheory__factor___reduced_totient():
    if False:
        return 10
    from sympy.ntheory.factor_ import reduced_totient
    k = symbols('k', integer=True)
    t = reduced_totient(k)
    assert _test_args(t)

def test_sympy__ntheory__factor___divisor_sigma():
    if False:
        return 10
    from sympy.ntheory.factor_ import divisor_sigma
    k = symbols('k', integer=True)
    n = symbols('n', integer=True)
    t = divisor_sigma(n, k)
    assert _test_args(t)

def test_sympy__ntheory__factor___udivisor_sigma():
    if False:
        print('Hello World!')
    from sympy.ntheory.factor_ import udivisor_sigma
    k = symbols('k', integer=True)
    n = symbols('n', integer=True)
    t = udivisor_sigma(n, k)
    assert _test_args(t)

def test_sympy__ntheory__factor___primenu():
    if False:
        i = 10
        return i + 15
    from sympy.ntheory.factor_ import primenu
    n = symbols('n', integer=True)
    t = primenu(n)
    assert _test_args(t)

def test_sympy__ntheory__factor___primeomega():
    if False:
        return 10
    from sympy.ntheory.factor_ import primeomega
    n = symbols('n', integer=True)
    t = primeomega(n)
    assert _test_args(t)

def test_sympy__ntheory__residue_ntheory__mobius():
    if False:
        print('Hello World!')
    from sympy.ntheory import mobius
    assert _test_args(mobius(2))

def test_sympy__ntheory__generate__primepi():
    if False:
        i = 10
        return i + 15
    from sympy.ntheory import primepi
    n = symbols('n')
    t = primepi(n)
    assert _test_args(t)

def test_sympy__physics__optics__waves__TWave():
    if False:
        for i in range(10):
            print('nop')
    from sympy.physics.optics import TWave
    (A, f, phi) = symbols('A, f, phi')
    assert _test_args(TWave(A, f, phi))

def test_sympy__physics__optics__gaussopt__BeamParameter():
    if False:
        while True:
            i = 10
    from sympy.physics.optics import BeamParameter
    assert _test_args(BeamParameter(5.3e-07, 1, w=0.001, n=1))

def test_sympy__physics__optics__medium__Medium():
    if False:
        return 10
    from sympy.physics.optics import Medium
    assert _test_args(Medium('m'))

def test_sympy__physics__optics__medium__MediumN():
    if False:
        for i in range(10):
            print('nop')
    from sympy.physics.optics.medium import Medium
    assert _test_args(Medium('m', n=2))

def test_sympy__physics__optics__medium__MediumPP():
    if False:
        while True:
            i = 10
    from sympy.physics.optics.medium import Medium
    assert _test_args(Medium('m', permittivity=2, permeability=2))

def test_sympy__tensor__array__expressions__array_expressions__ArrayContraction():
    if False:
        i = 10
        return i + 15
    from sympy.tensor.array.expressions.array_expressions import ArrayContraction
    from sympy.tensor.indexed import IndexedBase
    A = symbols('A', cls=IndexedBase)
    assert _test_args(ArrayContraction(A, (0, 1)))

def test_sympy__tensor__array__expressions__array_expressions__ArrayDiagonal():
    if False:
        print('Hello World!')
    from sympy.tensor.array.expressions.array_expressions import ArrayDiagonal
    from sympy.tensor.indexed import IndexedBase
    A = symbols('A', cls=IndexedBase)
    assert _test_args(ArrayDiagonal(A, (0, 1)))

def test_sympy__tensor__array__expressions__array_expressions__ArrayTensorProduct():
    if False:
        for i in range(10):
            print('nop')
    from sympy.tensor.array.expressions.array_expressions import ArrayTensorProduct
    from sympy.tensor.indexed import IndexedBase
    (A, B) = symbols('A B', cls=IndexedBase)
    assert _test_args(ArrayTensorProduct(A, B))

def test_sympy__tensor__array__expressions__array_expressions__ArrayAdd():
    if False:
        for i in range(10):
            print('nop')
    from sympy.tensor.array.expressions.array_expressions import ArrayAdd
    from sympy.tensor.indexed import IndexedBase
    (A, B) = symbols('A B', cls=IndexedBase)
    assert _test_args(ArrayAdd(A, B))

def test_sympy__tensor__array__expressions__array_expressions__PermuteDims():
    if False:
        print('Hello World!')
    from sympy.tensor.array.expressions.array_expressions import PermuteDims
    A = MatrixSymbol('A', 4, 4)
    assert _test_args(PermuteDims(A, (1, 0)))

def test_sympy__tensor__array__expressions__array_expressions__ArrayElementwiseApplyFunc():
    if False:
        while True:
            i = 10
    from sympy.tensor.array.expressions.array_expressions import ArraySymbol, ArrayElementwiseApplyFunc
    A = ArraySymbol('A', (4,))
    assert _test_args(ArrayElementwiseApplyFunc(exp, A))

def test_sympy__tensor__array__expressions__array_expressions__Reshape():
    if False:
        for i in range(10):
            print('nop')
    from sympy.tensor.array.expressions.array_expressions import ArraySymbol, Reshape
    A = ArraySymbol('A', (4,))
    assert _test_args(Reshape(A, (2, 2)))

def test_sympy__codegen__ast__Assignment():
    if False:
        return 10
    from sympy.codegen.ast import Assignment
    assert _test_args(Assignment(x, y))

def test_sympy__codegen__cfunctions__expm1():
    if False:
        i = 10
        return i + 15
    from sympy.codegen.cfunctions import expm1
    assert _test_args(expm1(x))

def test_sympy__codegen__cfunctions__log1p():
    if False:
        print('Hello World!')
    from sympy.codegen.cfunctions import log1p
    assert _test_args(log1p(x))

def test_sympy__codegen__cfunctions__exp2():
    if False:
        for i in range(10):
            print('nop')
    from sympy.codegen.cfunctions import exp2
    assert _test_args(exp2(x))

def test_sympy__codegen__cfunctions__log2():
    if False:
        while True:
            i = 10
    from sympy.codegen.cfunctions import log2
    assert _test_args(log2(x))

def test_sympy__codegen__cfunctions__fma():
    if False:
        print('Hello World!')
    from sympy.codegen.cfunctions import fma
    assert _test_args(fma(x, y, z))

def test_sympy__codegen__cfunctions__log10():
    if False:
        return 10
    from sympy.codegen.cfunctions import log10
    assert _test_args(log10(x))

def test_sympy__codegen__cfunctions__Sqrt():
    if False:
        for i in range(10):
            print('nop')
    from sympy.codegen.cfunctions import Sqrt
    assert _test_args(Sqrt(x))

def test_sympy__codegen__cfunctions__Cbrt():
    if False:
        return 10
    from sympy.codegen.cfunctions import Cbrt
    assert _test_args(Cbrt(x))

def test_sympy__codegen__cfunctions__hypot():
    if False:
        while True:
            i = 10
    from sympy.codegen.cfunctions import hypot
    assert _test_args(hypot(x, y))

def test_sympy__codegen__cfunctions__isnan():
    if False:
        i = 10
        return i + 15
    from sympy.codegen.cfunctions import isnan
    assert _test_args(isnan(x))

def test_sympy__codegen__fnodes__FFunction():
    if False:
        return 10
    from sympy.codegen.fnodes import FFunction
    assert _test_args(FFunction('f'))

def test_sympy__codegen__fnodes__F95Function():
    if False:
        i = 10
        return i + 15
    from sympy.codegen.fnodes import F95Function
    assert _test_args(F95Function('f'))

def test_sympy__codegen__fnodes__isign():
    if False:
        print('Hello World!')
    from sympy.codegen.fnodes import isign
    assert _test_args(isign(1, x))

def test_sympy__codegen__fnodes__dsign():
    if False:
        print('Hello World!')
    from sympy.codegen.fnodes import dsign
    assert _test_args(dsign(1, x))

def test_sympy__codegen__fnodes__cmplx():
    if False:
        for i in range(10):
            print('nop')
    from sympy.codegen.fnodes import cmplx
    assert _test_args(cmplx(x, y))

def test_sympy__codegen__fnodes__kind():
    if False:
        for i in range(10):
            print('nop')
    from sympy.codegen.fnodes import kind
    assert _test_args(kind(x))

def test_sympy__codegen__fnodes__merge():
    if False:
        return 10
    from sympy.codegen.fnodes import merge
    assert _test_args(merge(1, 2, Eq(x, 0)))

def test_sympy__codegen__fnodes___literal():
    if False:
        print('Hello World!')
    from sympy.codegen.fnodes import _literal
    assert _test_args(_literal(1))

def test_sympy__codegen__fnodes__literal_sp():
    if False:
        for i in range(10):
            print('nop')
    from sympy.codegen.fnodes import literal_sp
    assert _test_args(literal_sp(1))

def test_sympy__codegen__fnodes__literal_dp():
    if False:
        return 10
    from sympy.codegen.fnodes import literal_dp
    assert _test_args(literal_dp(1))

def test_sympy__codegen__matrix_nodes__MatrixSolve():
    if False:
        while True:
            i = 10
    from sympy.matrices import MatrixSymbol
    from sympy.codegen.matrix_nodes import MatrixSolve
    A = MatrixSymbol('A', 3, 3)
    v = MatrixSymbol('x', 3, 1)
    assert _test_args(MatrixSolve(A, v))

def test_sympy__vector__coordsysrect__CoordSys3D():
    if False:
        while True:
            i = 10
    from sympy.vector.coordsysrect import CoordSys3D
    assert _test_args(CoordSys3D('C'))

def test_sympy__vector__point__Point():
    if False:
        i = 10
        return i + 15
    from sympy.vector.point import Point
    assert _test_args(Point('P'))

def test_sympy__vector__basisdependent__BasisDependent():
    if False:
        for i in range(10):
            print('nop')
    pass

def test_sympy__vector__basisdependent__BasisDependentMul():
    if False:
        for i in range(10):
            print('nop')
    pass

def test_sympy__vector__basisdependent__BasisDependentAdd():
    if False:
        return 10
    pass

def test_sympy__vector__basisdependent__BasisDependentZero():
    if False:
        print('Hello World!')
    pass

def test_sympy__vector__vector__BaseVector():
    if False:
        while True:
            i = 10
    from sympy.vector.vector import BaseVector
    from sympy.vector.coordsysrect import CoordSys3D
    C = CoordSys3D('C')
    assert _test_args(BaseVector(0, C, ' ', ' '))

def test_sympy__vector__vector__VectorAdd():
    if False:
        print('Hello World!')
    from sympy.vector.vector import VectorAdd, VectorMul
    from sympy.vector.coordsysrect import CoordSys3D
    C = CoordSys3D('C')
    from sympy.abc import a, b, c, x, y, z
    v1 = a * C.i + b * C.j + c * C.k
    v2 = x * C.i + y * C.j + z * C.k
    assert _test_args(VectorAdd(v1, v2))
    assert _test_args(VectorMul(x, v1))

def test_sympy__vector__vector__VectorMul():
    if False:
        while True:
            i = 10
    from sympy.vector.vector import VectorMul
    from sympy.vector.coordsysrect import CoordSys3D
    C = CoordSys3D('C')
    from sympy.abc import a
    assert _test_args(VectorMul(a, C.i))

def test_sympy__vector__vector__VectorZero():
    if False:
        while True:
            i = 10
    from sympy.vector.vector import VectorZero
    assert _test_args(VectorZero())

def test_sympy__vector__vector__Vector():
    if False:
        while True:
            i = 10
    pass

def test_sympy__vector__vector__Cross():
    if False:
        while True:
            i = 10
    from sympy.vector.vector import Cross
    from sympy.vector.coordsysrect import CoordSys3D
    C = CoordSys3D('C')
    _test_args(Cross(C.i, C.j))

def test_sympy__vector__vector__Dot():
    if False:
        print('Hello World!')
    from sympy.vector.vector import Dot
    from sympy.vector.coordsysrect import CoordSys3D
    C = CoordSys3D('C')
    _test_args(Dot(C.i, C.j))

def test_sympy__vector__dyadic__Dyadic():
    if False:
        i = 10
        return i + 15
    pass

def test_sympy__vector__dyadic__BaseDyadic():
    if False:
        i = 10
        return i + 15
    from sympy.vector.dyadic import BaseDyadic
    from sympy.vector.coordsysrect import CoordSys3D
    C = CoordSys3D('C')
    assert _test_args(BaseDyadic(C.i, C.j))

def test_sympy__vector__dyadic__DyadicMul():
    if False:
        for i in range(10):
            print('nop')
    from sympy.vector.dyadic import BaseDyadic, DyadicMul
    from sympy.vector.coordsysrect import CoordSys3D
    C = CoordSys3D('C')
    assert _test_args(DyadicMul(3, BaseDyadic(C.i, C.j)))

def test_sympy__vector__dyadic__DyadicAdd():
    if False:
        while True:
            i = 10
    from sympy.vector.dyadic import BaseDyadic, DyadicAdd
    from sympy.vector.coordsysrect import CoordSys3D
    C = CoordSys3D('C')
    assert _test_args(2 * DyadicAdd(BaseDyadic(C.i, C.i), BaseDyadic(C.i, C.j)))

def test_sympy__vector__dyadic__DyadicZero():
    if False:
        print('Hello World!')
    from sympy.vector.dyadic import DyadicZero
    assert _test_args(DyadicZero())

def test_sympy__vector__deloperator__Del():
    if False:
        i = 10
        return i + 15
    from sympy.vector.deloperator import Del
    assert _test_args(Del())

def test_sympy__vector__implicitregion__ImplicitRegion():
    if False:
        for i in range(10):
            print('nop')
    from sympy.vector.implicitregion import ImplicitRegion
    from sympy.abc import x, y
    assert _test_args(ImplicitRegion((x, y), y ** 3 - 4 * x))

def test_sympy__vector__integrals__ParametricIntegral():
    if False:
        print('Hello World!')
    from sympy.vector.integrals import ParametricIntegral
    from sympy.vector.parametricregion import ParametricRegion
    from sympy.vector.coordsysrect import CoordSys3D
    C = CoordSys3D('C')
    assert _test_args(ParametricIntegral(C.y * C.i - 10 * C.j, ParametricRegion((x, y), (x, 1, 3), (y, -2, 2))))

def test_sympy__vector__operators__Curl():
    if False:
        print('Hello World!')
    from sympy.vector.operators import Curl
    from sympy.vector.coordsysrect import CoordSys3D
    C = CoordSys3D('C')
    assert _test_args(Curl(C.i))

def test_sympy__vector__operators__Laplacian():
    if False:
        return 10
    from sympy.vector.operators import Laplacian
    from sympy.vector.coordsysrect import CoordSys3D
    C = CoordSys3D('C')
    assert _test_args(Laplacian(C.i))

def test_sympy__vector__operators__Divergence():
    if False:
        return 10
    from sympy.vector.operators import Divergence
    from sympy.vector.coordsysrect import CoordSys3D
    C = CoordSys3D('C')
    assert _test_args(Divergence(C.i))

def test_sympy__vector__operators__Gradient():
    if False:
        for i in range(10):
            print('nop')
    from sympy.vector.operators import Gradient
    from sympy.vector.coordsysrect import CoordSys3D
    C = CoordSys3D('C')
    assert _test_args(Gradient(C.x))

def test_sympy__vector__orienters__Orienter():
    if False:
        return 10
    pass

def test_sympy__vector__orienters__ThreeAngleOrienter():
    if False:
        print('Hello World!')
    pass

def test_sympy__vector__orienters__AxisOrienter():
    if False:
        while True:
            i = 10
    from sympy.vector.orienters import AxisOrienter
    from sympy.vector.coordsysrect import CoordSys3D
    C = CoordSys3D('C')
    assert _test_args(AxisOrienter(x, C.i))

def test_sympy__vector__orienters__BodyOrienter():
    if False:
        i = 10
        return i + 15
    from sympy.vector.orienters import BodyOrienter
    assert _test_args(BodyOrienter(x, y, z, '123'))

def test_sympy__vector__orienters__SpaceOrienter():
    if False:
        i = 10
        return i + 15
    from sympy.vector.orienters import SpaceOrienter
    assert _test_args(SpaceOrienter(x, y, z, '123'))

def test_sympy__vector__orienters__QuaternionOrienter():
    if False:
        for i in range(10):
            print('nop')
    from sympy.vector.orienters import QuaternionOrienter
    (a, b, c, d) = symbols('a b c d')
    assert _test_args(QuaternionOrienter(a, b, c, d))

def test_sympy__vector__parametricregion__ParametricRegion():
    if False:
        print('Hello World!')
    from sympy.abc import t
    from sympy.vector.parametricregion import ParametricRegion
    assert _test_args(ParametricRegion((t, t ** 3), (t, 0, 2)))

def test_sympy__vector__scalar__BaseScalar():
    if False:
        while True:
            i = 10
    from sympy.vector.scalar import BaseScalar
    from sympy.vector.coordsysrect import CoordSys3D
    C = CoordSys3D('C')
    assert _test_args(BaseScalar(0, C, ' ', ' '))

def test_sympy__physics__wigner__Wigner3j():
    if False:
        print('Hello World!')
    from sympy.physics.wigner import Wigner3j
    assert _test_args(Wigner3j(0, 0, 0, 0, 0, 0))

def test_sympy__combinatorics__schur_number__SchurNumber():
    if False:
        for i in range(10):
            print('nop')
    from sympy.combinatorics.schur_number import SchurNumber
    assert _test_args(SchurNumber(x))

def test_sympy__combinatorics__perm_groups__SymmetricPermutationGroup():
    if False:
        while True:
            i = 10
    from sympy.combinatorics.perm_groups import SymmetricPermutationGroup
    assert _test_args(SymmetricPermutationGroup(5))

def test_sympy__combinatorics__perm_groups__Coset():
    if False:
        print('Hello World!')
    from sympy.combinatorics.permutations import Permutation
    from sympy.combinatorics.perm_groups import PermutationGroup, Coset
    a = Permutation(1, 2)
    b = Permutation(0, 1)
    G = PermutationGroup([a, b])
    assert _test_args(Coset(a, G))