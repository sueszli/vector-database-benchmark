"""Demonstration of quantum dense coding."""
from sympy import pprint
from sympy.physics.quantum import qapply
from sympy.physics.quantum.gate import H, X, Z, CNOT
from sympy.physics.quantum.grover import superposition_basis

def main():
    if False:
        while True:
            i = 10
    psi = superposition_basis(2)
    psi
    print('An even superposition of 2 qubits.  Assume Alice has the left QBit.')
    pprint(psi)
    print('To Send Bob the message |00>.')
    circuit = H(1) * CNOT(1, 0)
    result = qapply(circuit * psi)
    result
    pprint(result)
    print('To Send Bob the message |01>.')
    circuit = H(1) * CNOT(1, 0) * X(1)
    result = qapply(circuit * psi)
    result
    pprint(result)
    print('To Send Bob the message |10>.')
    circuit = H(1) * CNOT(1, 0) * Z(1)
    result = qapply(circuit * psi)
    result
    pprint(result)
    print('To Send Bob the message |11>.')
    circuit = H(1) * CNOT(1, 0) * Z(1) * X(1)
    result = qapply(circuit * psi)
    result
    pprint(result)
if __name__ == '__main__':
    main()