"""
Implementation of the GraySynth algorithm for synthesizing CNOT-Phase
circuits with efficient CNOT cost, and the Patel-Hayes-Markov algorithm
for optimal synthesis of linear (CNOT-only) reversible circuits.
"""
import copy
import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.exceptions import QiskitError
from qiskit.synthesis.linear import synth_cnot_count_full_pmh

def synth_cnot_phase_aam(cnots, angles, section_size=2):
    if False:
        i = 10
        return i + 15
    'This function is an implementation of the GraySynth algorithm of\n    Amy, Azimadeh and Mosca.\n\n    GraySynth is a heuristic algorithm from [1] for synthesizing small parity networks.\n    It is inspired by Gray codes. Given a set of binary strings S\n    (called "cnots" bellow), the algorithm synthesizes a parity network for S by\n    repeatedly choosing an index i to expand and then effectively recursing on\n    the co-factors S_0 and S_1, consisting of the strings y in S,\n    with y_i = 0 or 1 respectively. As a subset S is recursively expanded,\n    CNOT gates are applied so that a designated target bit contains the\n    (partial) parity ksi_y(x) where y_i = 1 if and only if y\'_i = 1 for all\n    y\' in S. If S is a singleton {y\'}, then y = y\', hence the target bit contains\n    the value ksi_y\'(x) as desired.\n\n    Notably, rather than uncomputing this sequence of CNOT gates when a subset S\n    is finished being synthesized, the algorithm maintains the invariant\n    that the remaining parities to be computed are expressed over the current state\n    of bits. This allows the algorithm to avoid the \'backtracking\' inherent in\n    uncomputing-based methods.\n\n    The algorithm is described in detail in section 4 of [1].\n\n    Args:\n        cnots (list[list]): a matrix whose columns are the parities to be synthesized\n            e.g.::\n\n                [[0, 1, 1, 1, 1, 1],\n                 [1, 0, 0, 1, 1, 1],\n                 [1, 0, 0, 1, 0, 0],\n                 [0, 0, 1, 0, 1, 0]]\n\n            corresponds to::\n\n                 x1^x2 + x0 + x0^x3 + x0^x1^x2 + x0^x1^x3 + x0^x1\n\n        angles (list): a list containing all the phase-shift gates which are\n            to be applied, in the same order as in "cnots". A number is\n            interpreted as the angle of p(angle), otherwise the elements\n            have to be \'t\', \'tdg\', \'s\', \'sdg\' or \'z\'.\n\n        section_size (int): the size of every section, used in _lwr_cnot_synth(), in the\n            Patel–Markov–Hayes algorithm. section_size must be a factor of num_qubits.\n\n    Returns:\n        QuantumCircuit: the decomposed quantum circuit.\n\n    Raises:\n        QiskitError: when dimensions of cnots and angles don\'t align.\n\n    References:\n        1. Matthew Amy, Parsiad Azimzadeh, and Michele Mosca.\n           *On the controlled-NOT complexity of controlled-NOT–phase circuits.*,\n           Quantum Science and Technology 4.1 (2018): 015002.\n           `arXiv:1712.01859 <https://arxiv.org/abs/1712.01859>`_\n    '
    num_qubits = len(cnots)
    qcir = QuantumCircuit(num_qubits)
    if len(cnots[0]) != len(angles):
        raise QiskitError('Size of "cnots" and "angles" do not match.')
    range_list = list(range(num_qubits))
    epsilon = num_qubits
    sta = []
    cnots_copy = np.transpose(np.array(copy.deepcopy(cnots)))
    state = np.eye(num_qubits).astype('int')
    for qubit in range(num_qubits):
        index = 0
        for icnots in cnots_copy:
            if np.array_equal(icnots, state[qubit]):
                if angles[index] == 't':
                    qcir.t(qubit)
                elif angles[index] == 'tdg':
                    qcir.tdg(qubit)
                elif angles[index] == 's':
                    qcir.s(qubit)
                elif angles[index] == 'sdg':
                    qcir.sdg(qubit)
                elif angles[index] == 'z':
                    qcir.z(qubit)
                else:
                    qcir.p(angles[index] % np.pi, qubit)
                del angles[index]
                cnots_copy = np.delete(cnots_copy, index, axis=0)
                if index == len(cnots_copy):
                    break
                index -= 1
            index += 1
    sta.append([cnots, range_list, epsilon])
    while sta != []:
        [cnots, ilist, qubit] = sta.pop()
        if cnots == []:
            continue
        if 0 <= qubit < num_qubits:
            condition = True
            while condition:
                condition = False
                for j in range(num_qubits):
                    if j != qubit and sum(cnots[j]) == len(cnots[j]):
                        condition = True
                        qcir.cx(j, qubit)
                        state[qubit] ^= state[j]
                        index = 0
                        for icnots in cnots_copy:
                            if np.array_equal(icnots, state[qubit]):
                                if angles[index] == 't':
                                    qcir.t(qubit)
                                elif angles[index] == 'tdg':
                                    qcir.tdg(qubit)
                                elif angles[index] == 's':
                                    qcir.s(qubit)
                                elif angles[index] == 'sdg':
                                    qcir.sdg(qubit)
                                elif angles[index] == 'z':
                                    qcir.z(qubit)
                                else:
                                    qcir.p(angles[index] % np.pi, qubit)
                                del angles[index]
                                cnots_copy = np.delete(cnots_copy, index, axis=0)
                                if index == len(cnots_copy):
                                    break
                                index -= 1
                            index += 1
                        for x in _remove_duplicates(sta + [[cnots, ilist, qubit]]):
                            [cnotsp, _, _] = x
                            if cnotsp == []:
                                continue
                            for ttt in range(len(cnotsp[j])):
                                cnotsp[j][ttt] ^= cnotsp[qubit][ttt]
        if ilist == []:
            continue
        j = ilist[np.argmax([[max(row.count(0), row.count(1)) for row in cnots][k] for k in ilist])]
        cnots0 = []
        cnots1 = []
        for y in list(map(list, zip(*cnots))):
            if y[j] == 0:
                cnots0.append(y)
            elif y[j] == 1:
                cnots1.append(y)
        cnots0 = list(map(list, zip(*cnots0)))
        cnots1 = list(map(list, zip(*cnots1)))
        if qubit == epsilon:
            sta.append([cnots1, list(set(ilist).difference([j])), j])
        else:
            sta.append([cnots1, list(set(ilist).difference([j])), qubit])
        sta.append([cnots0, list(set(ilist).difference([j])), qubit])
    qcir &= synth_cnot_count_full_pmh(state, section_size).inverse()
    return qcir

def _remove_duplicates(lists):
    if False:
        i = 10
        return i + 15
    '\n    Remove duplicates in list\n\n    Args:\n        lists (list): a list which may contain duplicate elements.\n\n    Returns:\n        list: a list which contains only unique elements.\n    '
    unique_list = []
    for element in lists:
        if element not in unique_list:
            unique_list.append(element)
    return unique_list