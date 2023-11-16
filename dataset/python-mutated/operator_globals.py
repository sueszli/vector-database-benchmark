"""
Operator Globals
"""
import warnings
from qiskit.quantum_info import Pauli
from qiskit.circuit.library import CXGate, SGate, TGate, HGate, SwapGate, CZGate
from qiskit.opflow.primitive_ops.pauli_op import PauliOp
from qiskit.opflow.primitive_ops.circuit_op import CircuitOp
from qiskit.opflow.state_fns.dict_state_fn import DictStateFn
from qiskit.utils.deprecation import deprecate_func
EVAL_SIG_DIGITS = 18

@deprecate_func(since='0.24.0', package_name='qiskit-terra', additional_msg='For code migration guidelines, visit https://qisk.it/opflow_migration.')
def make_immutable(obj):
    if False:
        print('Hello World!')
    'Deprecate\\: Delete the __setattr__ property to make the object mostly immutable.'
    obj.__setattr__ = None
    return obj
with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=DeprecationWarning, module='qiskit\\.opflow\\.')
    X = make_immutable(PauliOp(Pauli('X')))
    Y = make_immutable(PauliOp(Pauli('Y')))
    Z = make_immutable(PauliOp(Pauli('Z')))
    I = make_immutable(PauliOp(Pauli('I')))
    CX = make_immutable(CircuitOp(CXGate()))
    S = make_immutable(CircuitOp(SGate()))
    H = make_immutable(CircuitOp(HGate()))
    T = make_immutable(CircuitOp(TGate()))
    Swap = make_immutable(CircuitOp(SwapGate()))
    CZ = make_immutable(CircuitOp(CZGate()))
    Zero = make_immutable(DictStateFn('0'))
    One = make_immutable(DictStateFn('1'))
    Plus = make_immutable(H.compose(Zero))
    Minus = make_immutable(H.compose(X).compose(Zero))