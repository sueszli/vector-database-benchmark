"""Tests LogicNetwork.simulate method."""
import unittest
from ddt import ddt, data
from qiskit.test import QiskitTestCase
from qiskit.utils.optionals import HAS_TWEEDLEDUM
from . import utils
if HAS_TWEEDLEDUM:
    from qiskit.circuit.classicalfunction import classical_function as compile_classical_function

@unittest.skipUnless(HAS_TWEEDLEDUM, 'Tweedledum is required for these tests.')
@ddt
class TestSimulate(QiskitTestCase):
    """Tests LogicNetwork.simulate method"""

    @data(*utils.example_list())
    def test_(self, a_callable):
        if False:
            while True:
                i = 10
        'Tests LogicSimulate.simulate() on all the examples'
        network = compile_classical_function(a_callable)
        truth_table = network.simulate_all()
        self.assertEqual(truth_table, utils.get_truthtable_from_function(a_callable))