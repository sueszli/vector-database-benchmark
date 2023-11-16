"""Test adder circuits."""
import unittest
import numpy as np
from ddt import ddt, data, unpack
from qiskit.test.base import QiskitTestCase
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import CDKMRippleCarryAdder, DraperQFTAdder, VBERippleCarryAdder

@ddt
class TestAdder(QiskitTestCase):
    """Test the adder circuits."""

    def assertAdditionIsCorrect(self, num_state_qubits: int, adder: QuantumCircuit, inplace: bool, kind: str):
        if False:
            while True:
                i = 10
        'Assert that adder correctly implements the summation.\n\n        This test prepares a equal superposition state in both input registers, then performs\n        the addition on the superposition and checks that the output state is the expected\n        superposition of all possible additions.\n\n        Args:\n            num_state_qubits: The number of bits in the numbers that are added.\n            adder: The circuit performing the addition of two numbers with ``num_state_qubits``\n                bits.\n            inplace: If True, compare against an inplace addition where the result is written into\n                the second register plus carry qubit. If False, assume that the result is written\n                into a third register of appropriate size.\n            kind: TODO\n        '
        circuit = QuantumCircuit(*adder.qregs)
        if kind == 'full':
            num_superpos_qubits = 2 * num_state_qubits + 1
        else:
            num_superpos_qubits = 2 * num_state_qubits
        circuit.h(range(num_superpos_qubits))
        circuit.compose(adder, inplace=True)
        statevector = Statevector(circuit)
        probabilities = statevector.probabilities()
        pad = '0' * circuit.num_ancillas
        expectations = np.zeros_like(probabilities)
        num_bits_sum = num_state_qubits + 1
        for x in range(2 ** num_state_qubits):
            for y in range(2 ** num_state_qubits):
                if kind == 'full':
                    additions = [x + y, 1 + x + y]
                elif kind == 'half':
                    additions = [x + y]
                else:
                    additions = [(x + y) % 2 ** num_state_qubits]
                bin_x = bin(x)[2:].zfill(num_state_qubits)
                bin_y = bin(y)[2:].zfill(num_state_qubits)
                for (i, addition) in enumerate(additions):
                    bin_res = bin(addition)[2:].zfill(num_bits_sum)
                    if kind == 'full':
                        cin = str(i)
                        bin_index = pad + bin_res + bin_x + cin if inplace else pad + bin_res + bin_y + bin_x + cin
                    else:
                        bin_index = pad + bin_res + bin_x if inplace else pad + bin_res + bin_y + bin_x
                    index = int(bin_index, 2)
                    expectations[index] += 1 / 2 ** num_superpos_qubits
        np.testing.assert_array_almost_equal(expectations, probabilities)

    @data((3, CDKMRippleCarryAdder, True), (5, CDKMRippleCarryAdder, True), (3, CDKMRippleCarryAdder, True, 'fixed'), (5, CDKMRippleCarryAdder, True, 'fixed'), (1, CDKMRippleCarryAdder, True, 'full'), (3, CDKMRippleCarryAdder, True, 'full'), (5, CDKMRippleCarryAdder, True, 'full'), (3, DraperQFTAdder, True), (5, DraperQFTAdder, True), (3, DraperQFTAdder, True, 'fixed'), (5, DraperQFTAdder, True, 'fixed'), (1, VBERippleCarryAdder, True, 'full'), (3, VBERippleCarryAdder, True, 'full'), (5, VBERippleCarryAdder, True, 'full'), (1, VBERippleCarryAdder, True), (2, VBERippleCarryAdder, True), (5, VBERippleCarryAdder, True), (1, VBERippleCarryAdder, True, 'fixed'), (2, VBERippleCarryAdder, True, 'fixed'), (4, VBERippleCarryAdder, True, 'fixed'))
    @unpack
    def test_summation(self, num_state_qubits, adder, inplace, kind='half'):
        if False:
            i = 10
            return i + 15
        'Test summation for all implemented adders.'
        adder = adder(num_state_qubits, kind=kind)
        self.assertAdditionIsCorrect(num_state_qubits, adder, inplace, kind)

    @data(CDKMRippleCarryAdder, DraperQFTAdder, VBERippleCarryAdder)
    def test_raises_on_wrong_num_bits(self, adder):
        if False:
            return 10
        'Test an error is raised for a bad number of qubits.'
        with self.assertRaises(ValueError):
            _ = adder(-1)
if __name__ == '__main__':
    unittest.main()