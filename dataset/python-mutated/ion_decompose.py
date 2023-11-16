"""
Decomposition methods for trapped-ion basis gates RXXGate, RXGate, RYGate.
"""
from __future__ import annotations
import numpy as np
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.library.standard_gates.ry import RYGate
from qiskit.circuit.library.standard_gates.rx import RXGate
from qiskit.circuit.library.standard_gates.rxx import RXXGate

def cnot_rxx_decompose(plus_ry: bool=True, plus_rxx: bool=True):
    if False:
        print('Hello World!')
    'Decomposition of CNOT gate.\n\n    NOTE: this differs to CNOT by a global phase.\n    The matrix returned is given by exp(1j * pi/4) * CNOT\n\n    Args:\n        plus_ry (bool): positive initial RY rotation\n        plus_rxx (bool): positive RXX rotation.\n\n    Returns:\n        QuantumCircuit: The decomposed circuit for CNOT gate (up to\n        global phase).\n    '
    if plus_ry:
        sgn_ry = 1
    else:
        sgn_ry = -1
    if plus_rxx:
        sgn_rxx = 1
    else:
        sgn_rxx = -1
    circuit = QuantumCircuit(2, global_phase=-sgn_ry * sgn_rxx * np.pi / 4)
    circuit.append(RYGate(sgn_ry * np.pi / 2), [0])
    circuit.append(RXXGate(sgn_rxx * np.pi / 2), [0, 1])
    circuit.append(RXGate(-sgn_rxx * np.pi / 2), [0])
    circuit.append(RXGate(-sgn_rxx * sgn_ry * np.pi / 2), [1])
    circuit.append(RYGate(-sgn_ry * np.pi / 2), [0])
    return circuit