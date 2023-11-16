"""
Simple circuit constructors for Weyl reflections.
"""
from __future__ import annotations
import numpy as np
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.library.standard_gates import RXGate, RYGate, RZGate
reflection_options = {'no reflection': ([1, 1, 1], 1, []), 'reflect XX, YY': ([-1, -1, 1], 1, [RZGate]), 'reflect XX, ZZ': ([-1, 1, -1], 1, [RYGate]), 'reflect YY, ZZ': ([1, -1, -1], 1, [RXGate])}
'\nA table of available reflection transformations on canonical coordinates.\nEntries take the form\n\n    readable_name: (reflection scalars, global phase, [gate constructors]),\n\nwhere reflection scalars (a, b, c) model the map (x, y, z) |-> (ax, by, cz),\nglobal phase is a complex unit, and gate constructors are applied in sequence\nand by conjugation to the first qubit and are passed pi as a parameter.\n'
shift_options = {'no shift': ([0, 0, 0], 1, []), 'Z shift': ([0, 0, 1], 1j, [RZGate]), 'Y shift': ([0, 1, 0], -1j, [RYGate]), 'Y,Z shift': ([0, 1, 1], 1, [RYGate, RZGate]), 'X shift': ([1, 0, 0], -1j, [RXGate]), 'X,Z shift': ([1, 0, 1], 1, [RXGate, RZGate]), 'X,Y shift': ([1, 1, 0], -1, [RXGate, RYGate]), 'X,Y,Z shift': ([1, 1, 1], -1j, [RXGate, RYGate, RZGate])}
'\nA table of available shift transformations on canonical coordinates.  Entries\ntake the form\n\n    readable name: (shift scalars, global phase, [gate constructors]),\n\nwhere shift scalars model the map\n\n    (x, y, z) |-> (x + a pi / 2, y + b pi / 2, z + c pi / 2) ,\n\nglobal phase is a complex unit, and gate constructors are applied to the first\nand second qubits and are passed pi as a parameter.\n'

def apply_reflection(reflection_name, coordinate):
    if False:
        while True:
            i = 10
    '\n    Given a reflection type and a canonical coordinate, applies the reflection\n    and describes a circuit which enacts the reflection + a global phase shift.\n    '
    (reflection_scalars, reflection_phase_shift, source_reflection_gates) = reflection_options[reflection_name]
    reflected_coord = [x * y for (x, y) in zip(reflection_scalars, coordinate)]
    source_reflection = QuantumCircuit(2)
    for gate in source_reflection_gates:
        source_reflection.append(gate(np.pi), [0])
    return (reflected_coord, source_reflection, reflection_phase_shift)

def apply_shift(shift_name, coordinate):
    if False:
        for i in range(10):
            print('nop')
    '\n    Given a shift type and a canonical coordinate, applies the shift and\n    describes a circuit which enacts the shift + a global phase shift.\n    '
    (shift_scalars, shift_phase_shift, source_shift_gates) = shift_options[shift_name]
    shifted_coord = [np.pi / 2 * x + y for (x, y) in zip(shift_scalars, coordinate)]
    source_shift = QuantumCircuit(2)
    for gate in source_shift_gates:
        source_shift.append(gate(np.pi), [0])
        source_shift.append(gate(np.pi), [1])
    return (shifted_coord, source_shift, shift_phase_shift)

def canonical_rotation_circuit(first_index, second_index):
    if False:
        return 10
    '\n    Given a pair of distinct indices 0 ≤ (first_index, second_index) ≤ 2,\n    produces a two-qubit circuit which rotates a canonical gate\n\n        a0 XX + a1 YY + a2 ZZ\n\n    into\n\n        a[first] XX + a[second] YY + a[other] ZZ .\n    '
    conj = QuantumCircuit(2)
    if (0, 1) == (first_index, second_index):
        pass
    elif (0, 2) == (first_index, second_index):
        conj.rx(-np.pi / 2, [0])
        conj.rx(np.pi / 2, [1])
    elif (1, 0) == (first_index, second_index):
        conj.rz(-np.pi / 2, [0])
        conj.rz(-np.pi / 2, [1])
    elif (1, 2) == (first_index, second_index):
        conj.rz(np.pi / 2, [0])
        conj.rz(np.pi / 2, [1])
        conj.ry(np.pi / 2, [0])
        conj.ry(-np.pi / 2, [1])
    elif (2, 0) == (first_index, second_index):
        conj.rz(np.pi / 2, [0])
        conj.rz(np.pi / 2, [1])
        conj.rx(np.pi / 2, [0])
        conj.rx(-np.pi / 2, [1])
    elif (2, 1) == (first_index, second_index):
        conj.ry(np.pi / 2, [0])
        conj.ry(-np.pi / 2, [1])
    return conj