""" A module for mapping operators to their corresponding eigenstates
and vice versa

It contains a global dictionary with eigenstate-operator pairings.
If a new state-operator pair is created, this dictionary should be
updated as well.

It also contains functions operators_to_state and state_to_operators
for mapping between the two. These can handle both classes and
instances of operators and states. See the individual function
descriptions for details.

TODO List:
- Update the dictionary with a complete list of state-operator pairs
"""
from sympy.physics.quantum.cartesian import XOp, YOp, ZOp, XKet, PxOp, PxKet, PositionKet3D
from sympy.physics.quantum.operator import Operator
from sympy.physics.quantum.state import StateBase, BraBase, Ket
from sympy.physics.quantum.spin import JxOp, JyOp, JzOp, J2Op, JxKet, JyKet, JzKet
__all__ = ['operators_to_state', 'state_to_operators']
state_mapping = {JxKet: frozenset((J2Op, JxOp)), JyKet: frozenset((J2Op, JyOp)), JzKet: frozenset((J2Op, JzOp)), Ket: Operator, PositionKet3D: frozenset((XOp, YOp, ZOp)), PxKet: PxOp, XKet: XOp}
op_mapping = {v: k for (k, v) in state_mapping.items()}

def operators_to_state(operators, **options):
    if False:
        i = 10
        return i + 15
    ' Returns the eigenstate of the given operator or set of operators\n\n    A global function for mapping operator classes to their associated\n    states. It takes either an Operator or a set of operators and\n    returns the state associated with these.\n\n    This function can handle both instances of a given operator or\n    just the class itself (i.e. both XOp() and XOp)\n\n    There are multiple use cases to consider:\n\n    1) A class or set of classes is passed: First, we try to\n    instantiate default instances for these operators. If this fails,\n    then the class is simply returned. If we succeed in instantiating\n    default instances, then we try to call state._operators_to_state\n    on the operator instances. If this fails, the class is returned.\n    Otherwise, the instance returned by _operators_to_state is returned.\n\n    2) An instance or set of instances is passed: In this case,\n    state._operators_to_state is called on the instances passed. If\n    this fails, a state class is returned. If the method returns an\n    instance, that instance is returned.\n\n    In both cases, if the operator class or set does not exist in the\n    state_mapping dictionary, None is returned.\n\n    Parameters\n    ==========\n\n    arg: Operator or set\n         The class or instance of the operator or set of operators\n         to be mapped to a state\n\n    Examples\n    ========\n\n    >>> from sympy.physics.quantum.cartesian import XOp, PxOp\n    >>> from sympy.physics.quantum.operatorset import operators_to_state\n    >>> from sympy.physics.quantum.operator import Operator\n    >>> operators_to_state(XOp)\n    |x>\n    >>> operators_to_state(XOp())\n    |x>\n    >>> operators_to_state(PxOp)\n    |px>\n    >>> operators_to_state(PxOp())\n    |px>\n    >>> operators_to_state(Operator)\n    |psi>\n    >>> operators_to_state(Operator())\n    |psi>\n    '
    if not (isinstance(operators, (Operator, set)) or issubclass(operators, Operator)):
        raise NotImplementedError('Argument is not an Operator or a set!')
    if isinstance(operators, set):
        for s in operators:
            if not (isinstance(s, Operator) or issubclass(s, Operator)):
                raise NotImplementedError('Set is not all Operators!')
        ops = frozenset(operators)
        if ops in op_mapping:
            try:
                op_instances = [op() for op in ops]
                ret = _get_state(op_mapping[ops], set(op_instances), **options)
            except NotImplementedError:
                ret = op_mapping[ops]
            return ret
        else:
            tmp = [type(o) for o in ops]
            classes = frozenset(tmp)
            if classes in op_mapping:
                ret = _get_state(op_mapping[classes], ops, **options)
            else:
                ret = None
            return ret
    elif operators in op_mapping:
        try:
            op_instance = operators()
            ret = _get_state(op_mapping[operators], op_instance, **options)
        except NotImplementedError:
            ret = op_mapping[operators]
        return ret
    elif type(operators) in op_mapping:
        return _get_state(op_mapping[type(operators)], operators, **options)
    else:
        return None

def state_to_operators(state, **options):
    if False:
        print('Hello World!')
    " Returns the operator or set of operators corresponding to the\n    given eigenstate\n\n    A global function for mapping state classes to their associated\n    operators or sets of operators. It takes either a state class\n    or instance.\n\n    This function can handle both instances of a given state or just\n    the class itself (i.e. both XKet() and XKet)\n\n    There are multiple use cases to consider:\n\n    1) A state class is passed: In this case, we first try\n    instantiating a default instance of the class. If this succeeds,\n    then we try to call state._state_to_operators on that instance.\n    If the creation of the default instance or if the calling of\n    _state_to_operators fails, then either an operator class or set of\n    operator classes is returned. Otherwise, the appropriate\n    operator instances are returned.\n\n    2) A state instance is returned: Here, state._state_to_operators\n    is called for the instance. If this fails, then a class or set of\n    operator classes is returned. Otherwise, the instances are returned.\n\n    In either case, if the state's class does not exist in\n    state_mapping, None is returned.\n\n    Parameters\n    ==========\n\n    arg: StateBase class or instance (or subclasses)\n         The class or instance of the state to be mapped to an\n         operator or set of operators\n\n    Examples\n    ========\n\n    >>> from sympy.physics.quantum.cartesian import XKet, PxKet, XBra, PxBra\n    >>> from sympy.physics.quantum.operatorset import state_to_operators\n    >>> from sympy.physics.quantum.state import Ket, Bra\n    >>> state_to_operators(XKet)\n    X\n    >>> state_to_operators(XKet())\n    X\n    >>> state_to_operators(PxKet)\n    Px\n    >>> state_to_operators(PxKet())\n    Px\n    >>> state_to_operators(PxBra)\n    Px\n    >>> state_to_operators(XBra)\n    X\n    >>> state_to_operators(Ket)\n    O\n    >>> state_to_operators(Bra)\n    O\n    "
    if not (isinstance(state, StateBase) or issubclass(state, StateBase)):
        raise NotImplementedError('Argument is not a state!')
    if state in state_mapping:
        state_inst = _make_default(state)
        try:
            ret = _get_ops(state_inst, _make_set(state_mapping[state]), **options)
        except (NotImplementedError, TypeError):
            ret = state_mapping[state]
    elif type(state) in state_mapping:
        ret = _get_ops(state, _make_set(state_mapping[type(state)]), **options)
    elif isinstance(state, BraBase) and state.dual_class() in state_mapping:
        ret = _get_ops(state, _make_set(state_mapping[state.dual_class()]))
    elif issubclass(state, BraBase) and state.dual_class() in state_mapping:
        state_inst = _make_default(state)
        try:
            ret = _get_ops(state_inst, _make_set(state_mapping[state.dual_class()]))
        except (NotImplementedError, TypeError):
            ret = state_mapping[state.dual_class()]
    else:
        ret = None
    return _make_set(ret)

def _make_default(expr):
    if False:
        i = 10
        return i + 15
    try:
        ret = expr()
    except TypeError:
        ret = expr
    return ret

def _get_state(state_class, ops, **options):
    if False:
        for i in range(10):
            print('nop')
    try:
        ret = state_class._operators_to_state(ops, **options)
    except NotImplementedError:
        ret = _make_default(state_class)
    return ret

def _get_ops(state_inst, op_classes, **options):
    if False:
        print('Hello World!')
    try:
        ret = state_inst._state_to_operators(op_classes, **options)
    except NotImplementedError:
        if isinstance(op_classes, (set, tuple, frozenset)):
            ret = tuple((_make_default(x) for x in op_classes))
        else:
            ret = _make_default(op_classes)
    if isinstance(ret, set) and len(ret) == 1:
        return ret[0]
    return ret

def _make_set(ops):
    if False:
        return 10
    if isinstance(ops, (tuple, list, frozenset)):
        return set(ops)
    else:
        return ops