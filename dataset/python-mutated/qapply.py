"""Logic for applying operators to states.

Todo:
* Sometimes the final result needs to be expanded, we should do this by hand.
"""
from sympy.core.add import Add
from sympy.core.mul import Mul
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.sympify import sympify
from sympy.physics.quantum.anticommutator import AntiCommutator
from sympy.physics.quantum.commutator import Commutator
from sympy.physics.quantum.dagger import Dagger
from sympy.physics.quantum.innerproduct import InnerProduct
from sympy.physics.quantum.operator import OuterProduct, Operator
from sympy.physics.quantum.state import State, KetBase, BraBase, Wavefunction
from sympy.physics.quantum.tensorproduct import TensorProduct
__all__ = ['qapply']

def qapply(e, **options):
    if False:
        return 10
    "Apply operators to states in a quantum expression.\n\n    Parameters\n    ==========\n\n    e : Expr\n        The expression containing operators and states. This expression tree\n        will be walked to find operators acting on states symbolically.\n    options : dict\n        A dict of key/value pairs that determine how the operator actions\n        are carried out.\n\n        The following options are valid:\n\n        * ``dagger``: try to apply Dagger operators to the left\n          (default: False).\n        * ``ip_doit``: call ``.doit()`` in inner products when they are\n          encountered (default: True).\n\n    Returns\n    =======\n\n    e : Expr\n        The original expression, but with the operators applied to states.\n\n    Examples\n    ========\n\n        >>> from sympy.physics.quantum import qapply, Ket, Bra\n        >>> b = Bra('b')\n        >>> k = Ket('k')\n        >>> A = k * b\n        >>> A\n        |k><b|\n        >>> qapply(A * b.dual / (b * b.dual))\n        |k>\n        >>> qapply(k.dual * A / (k.dual * k), dagger=True)\n        <b|\n        >>> qapply(k.dual * A / (k.dual * k))\n        <k|*|k><b|/<k|k>\n    "
    from sympy.physics.quantum.density import Density
    dagger = options.get('dagger', False)
    if e == 0:
        return S.Zero
    e = e.expand(commutator=True, tensorproduct=True)
    if isinstance(e, KetBase):
        return e
    elif isinstance(e, Add):
        result = 0
        for arg in e.args:
            result += qapply(arg, **options)
        return result.expand()
    elif isinstance(e, Density):
        new_args = [(qapply(state, **options), prob) for (state, prob) in e.args]
        return Density(*new_args)
    elif isinstance(e, TensorProduct):
        return TensorProduct(*[qapply(t, **options) for t in e.args])
    elif isinstance(e, Pow):
        return qapply(e.base, **options) ** e.exp
    elif isinstance(e, Mul):
        (c_part, nc_part) = e.args_cnc()
        c_mul = Mul(*c_part)
        nc_mul = Mul(*nc_part)
        if isinstance(nc_mul, Mul):
            result = c_mul * qapply_Mul(nc_mul, **options)
        else:
            result = c_mul * qapply(nc_mul, **options)
        if result == e and dagger:
            return Dagger(qapply_Mul(Dagger(e), **options))
        else:
            return result
    else:
        return e

def qapply_Mul(e, **options):
    if False:
        return 10
    ip_doit = options.get('ip_doit', True)
    args = list(e.args)
    if len(args) <= 1 or not isinstance(e, Mul):
        return e
    rhs = args.pop()
    lhs = args.pop()
    if not isinstance(rhs, Wavefunction) and sympify(rhs).is_commutative or (not isinstance(lhs, Wavefunction) and sympify(lhs).is_commutative):
        return e
    if isinstance(lhs, Pow) and lhs.exp.is_Integer:
        args.append(lhs.base ** (lhs.exp - 1))
        lhs = lhs.base
    if isinstance(lhs, OuterProduct):
        args.append(lhs.ket)
        lhs = lhs.bra
    if isinstance(lhs, (Commutator, AntiCommutator)):
        comm = lhs.doit()
        if isinstance(comm, Add):
            return qapply(e.func(*args + [comm.args[0], rhs]) + e.func(*args + [comm.args[1], rhs]), **options)
        else:
            return qapply(e.func(*args) * comm * rhs, **options)
    if isinstance(lhs, TensorProduct) and all((isinstance(arg, (Operator, State, Mul, Pow)) or arg == 1 for arg in lhs.args)) and isinstance(rhs, TensorProduct) and all((isinstance(arg, (Operator, State, Mul, Pow)) or arg == 1 for arg in rhs.args)) and (len(lhs.args) == len(rhs.args)):
        result = TensorProduct(*[qapply(lhs.args[n] * rhs.args[n], **options) for n in range(len(lhs.args))]).expand(tensorproduct=True)
        return qapply_Mul(e.func(*args), **options) * result
    try:
        result = lhs._apply_operator(rhs, **options)
    except NotImplementedError:
        result = None
    if result is None:
        _apply_right = getattr(rhs, '_apply_from_right_to', None)
        if _apply_right is not None:
            try:
                result = _apply_right(lhs, **options)
            except NotImplementedError:
                result = None
    if result is None:
        if isinstance(lhs, BraBase) and isinstance(rhs, KetBase):
            result = InnerProduct(lhs, rhs)
            if ip_doit:
                result = result.doit()
    if result == 0:
        return S.Zero
    elif result is None:
        if len(args) == 0:
            return e
        else:
            return qapply_Mul(e.func(*args + [lhs]), **options) * rhs
    elif isinstance(result, InnerProduct):
        return result * qapply_Mul(e.func(*args), **options)
    else:
        return qapply(e.func(*args) * result, **options)