from sympy.functions.elementary.miscellaneous import sqrt
from sympy.matrices.dense import Matrix
from sympy.physics.quantum.represent import represent
from sympy.physics.quantum.qapply import qapply
from sympy.physics.quantum.qubit import IntQubit
from sympy.physics.quantum.grover import apply_grover, superposition_basis, OracleGate, grover_iteration, WGate

def return_one_on_two(qubits):
    if False:
        while True:
            i = 10
    return qubits == IntQubit(2, qubits.nqubits)

def return_one_on_one(qubits):
    if False:
        return 10
    return qubits == IntQubit(1, nqubits=qubits.nqubits)

def test_superposition_basis():
    if False:
        i = 10
        return i + 15
    nbits = 2
    first_half_state = IntQubit(0, nqubits=nbits) / 2 + IntQubit(1, nqubits=nbits) / 2
    second_half_state = IntQubit(2, nbits) / 2 + IntQubit(3, nbits) / 2
    assert first_half_state + second_half_state == superposition_basis(nbits)
    nbits = 3
    firstq = 1 / sqrt(8) * IntQubit(0, nqubits=nbits) + 1 / sqrt(8) * IntQubit(1, nqubits=nbits)
    secondq = 1 / sqrt(8) * IntQubit(2, nbits) + 1 / sqrt(8) * IntQubit(3, nbits)
    thirdq = 1 / sqrt(8) * IntQubit(4, nbits) + 1 / sqrt(8) * IntQubit(5, nbits)
    fourthq = 1 / sqrt(8) * IntQubit(6, nbits) + 1 / sqrt(8) * IntQubit(7, nbits)
    assert firstq + secondq + thirdq + fourthq == superposition_basis(nbits)

def test_OracleGate():
    if False:
        print('Hello World!')
    v = OracleGate(1, lambda qubits: qubits == IntQubit(0))
    assert qapply(v * IntQubit(0)) == -IntQubit(0)
    assert qapply(v * IntQubit(1)) == IntQubit(1)
    nbits = 2
    v = OracleGate(2, return_one_on_two)
    assert qapply(v * IntQubit(0, nbits)) == IntQubit(0, nqubits=nbits)
    assert qapply(v * IntQubit(1, nbits)) == IntQubit(1, nqubits=nbits)
    assert qapply(v * IntQubit(2, nbits)) == -IntQubit(2, nbits)
    assert qapply(v * IntQubit(3, nbits)) == IntQubit(3, nbits)
    assert represent(OracleGate(1, lambda qubits: qubits == IntQubit(0)), nqubits=1) == Matrix([[-1, 0], [0, 1]])
    assert represent(v, nqubits=2) == Matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

def test_WGate():
    if False:
        i = 10
        return i + 15
    nqubits = 2
    basis_states = superposition_basis(nqubits)
    assert qapply(WGate(nqubits) * basis_states) == basis_states
    expected = 2 / sqrt(pow(2, nqubits)) * basis_states - IntQubit(1, nqubits=nqubits)
    assert qapply(WGate(nqubits) * IntQubit(1, nqubits=nqubits)) == expected

def test_grover_iteration_1():
    if False:
        while True:
            i = 10
    numqubits = 2
    basis_states = superposition_basis(numqubits)
    v = OracleGate(numqubits, return_one_on_one)
    expected = IntQubit(1, nqubits=numqubits)
    assert qapply(grover_iteration(basis_states, v)) == expected

def test_grover_iteration_2():
    if False:
        print('Hello World!')
    numqubits = 4
    basis_states = superposition_basis(numqubits)
    v = OracleGate(numqubits, return_one_on_two)
    iterated = grover_iteration(basis_states, v)
    iterated = qapply(iterated)
    iterated = grover_iteration(iterated, v)
    iterated = qapply(iterated)
    iterated = grover_iteration(iterated, v)
    iterated = qapply(iterated)
    expected = -13 * basis_states / 64 + 264 * IntQubit(2, numqubits) / 256
    assert qapply(expected) == iterated

def test_grover():
    if False:
        for i in range(10):
            print('nop')
    nqubits = 2
    assert apply_grover(return_one_on_one, nqubits) == IntQubit(1, nqubits=nqubits)
    nqubits = 4
    basis_states = superposition_basis(nqubits)
    expected = -13 * basis_states / 64 + 264 * IntQubit(2, nqubits) / 256
    assert apply_grover(return_one_on_two, 4) == qapply(expected)