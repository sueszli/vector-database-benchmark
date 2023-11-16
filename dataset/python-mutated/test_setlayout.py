"""Test the SetLayout pass"""
import unittest
from qiskit import QuantumRegister, QuantumCircuit, ClassicalRegister
from qiskit.transpiler import CouplingMap, Layout
from qiskit.transpiler.passes import SetLayout, ApplyLayout, FullAncillaAllocation
from qiskit.test import QiskitTestCase
from qiskit.transpiler import PassManager, TranspilerError

class TestSetLayout(QiskitTestCase):
    """Tests the SetLayout pass"""

    def assertEqualToReference(self, result_to_compare):
        if False:
            while True:
                i = 10
        'Compare result_to_compare to a reference\n\n                       ┌───┐ ░ ┌─┐\n              q_0 -> 0 ┤ H ├─░─┤M├───────────────\n                       ├───┤ ░ └╥┘┌─┐\n              q_1 -> 1 ┤ H ├─░──╫─┤M├────────────\n                       ├───┤ ░  ║ └╥┘      ┌─┐\n              q_4 -> 2 ┤ H ├─░──╫──╫───────┤M├───\n                       ├───┤ ░  ║  ║ ┌─┐   └╥┘\n              q_2 -> 3 ┤ H ├─░──╫──╫─┤M├────╫────\n                       └───┘ ░  ║  ║ └╥┘    ║\n        ancilla_0 -> 4 ─────────╫──╫──╫─────╫────\n                       ┌───┐ ░  ║  ║  ║ ┌─┐ ║\n              q_3 -> 5 ┤ H ├─░──╫──╫──╫─┤M├─╫────\n                       ├───┤ ░  ║  ║  ║ └╥┘ ║ ┌─┐\n              q_5 -> 6 ┤ H ├─░──╫──╫──╫──╫──╫─┤M├\n                       └───┘ ░  ║  ║  ║  ║  ║ └╥┘\n               meas: 6/═════════╩══╩══╩══╩══╩══╩═\n                                0  1  2  3  4  5\n        '
        qr = QuantumRegister(6, 'q')
        ancilla = QuantumRegister(1, 'ancilla')
        cl = ClassicalRegister(6, 'meas')
        reference = QuantumCircuit(qr, ancilla, cl)
        reference.h(qr)
        reference.barrier(qr)
        reference.measure(qr, cl)
        pass_manager = PassManager()
        pass_manager.append(SetLayout(Layout({qr[0]: 0, qr[1]: 1, qr[4]: 2, qr[2]: 3, ancilla[0]: 4, qr[3]: 5, qr[5]: 6})))
        pass_manager.append(ApplyLayout())
        self.assertEqual(result_to_compare, pass_manager.run(reference))

    def test_setlayout_as_Layout(self):
        if False:
            i = 10
            return i + 15
        'Construct SetLayout with a Layout.'
        qr = QuantumRegister(6, 'q')
        circuit = QuantumCircuit(qr)
        circuit.h(qr)
        circuit.measure_all()
        pass_manager = PassManager()
        pass_manager.append(SetLayout(Layout.from_intlist([0, 1, 3, 5, 2, 6], QuantumRegister(6, 'q'))))
        pass_manager.append(FullAncillaAllocation(CouplingMap.from_line(7)))
        pass_manager.append(ApplyLayout())
        result = pass_manager.run(circuit)
        self.assertEqualToReference(result)

    def test_setlayout_as_list(self):
        if False:
            i = 10
            return i + 15
        'Construct SetLayout with a list.'
        qr = QuantumRegister(6, 'q')
        circuit = QuantumCircuit(qr)
        circuit.h(qr)
        circuit.measure_all()
        pass_manager = PassManager()
        pass_manager.append(SetLayout([0, 1, 3, 5, 2, 6]))
        pass_manager.append(FullAncillaAllocation(CouplingMap.from_line(7)))
        pass_manager.append(ApplyLayout())
        result = pass_manager.run(circuit)
        self.assertEqualToReference(result)

    def test_raise_when_layout_len_does_not_match(self):
        if False:
            return 10
        'Test error is raised if layout defined as list does not match the circuit size.'
        qr = QuantumRegister(42, 'q')
        circuit = QuantumCircuit(qr)
        pass_manager = PassManager()
        pass_manager.append(SetLayout([0, 1, 3, 5, 2, 6]))
        pass_manager.append(FullAncillaAllocation(CouplingMap.from_line(7)))
        pass_manager.append(ApplyLayout())
        with self.assertRaises(TranspilerError):
            pass_manager.run(circuit)
if __name__ == '__main__':
    unittest.main()