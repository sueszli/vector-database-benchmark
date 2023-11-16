"""ClassicalFunction class"""
import ast
from typing import Callable, Optional
from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiskit.exceptions import QiskitError
from qiskit.utils.optionals import HAS_TWEEDLEDUM
from .classical_element import ClassicalElement
from .classical_function_visitor import ClassicalFunctionVisitor
from .utils import tweedledum2qiskit

@HAS_TWEEDLEDUM.require_in_instance
class ClassicalFunction(ClassicalElement):
    """Represent a classical function and its logic network."""

    def __init__(self, source, name=None):
        if False:
            for i in range(10):
                print('nop')
        'Creates a ``ClassicalFunction`` from Python source code in ``source``.\n\n        The code should be a single function with types.\n\n        Args:\n            source (str): Python code with type hints.\n            name (str): Optional. Default: "*classicalfunction*". ClassicalFunction name.\n\n        Raises:\n            QiskitError: If source is not a string.\n        '
        if not isinstance(source, str):
            raise QiskitError('ClassicalFunction needs a source code as a string.')
        self._ast = ast.parse(source)
        self._network = None
        self._scopes = None
        self._args = None
        self._truth_table = None
        super().__init__(name or '*classicalfunction*', num_qubits=sum((qreg.size for qreg in self.qregs)), params=[])

    def compile(self):
        if False:
            while True:
                i = 10
        'Parses and creates the logical circuit'
        _classical_function_visitor = ClassicalFunctionVisitor()
        _classical_function_visitor.visit(self._ast)
        self._network = _classical_function_visitor._network
        self._scopes = _classical_function_visitor.scopes
        self._args = _classical_function_visitor.args
        self.name = _classical_function_visitor.name

    @property
    def network(self):
        if False:
            for i in range(10):
                print('nop')
        'Returns the logical network'
        if self._network is None:
            self.compile()
        return self._network

    @property
    def scopes(self):
        if False:
            return 10
        'Returns the scope dict'
        if self._scopes is None:
            self.compile()
        return self._scopes

    @property
    def args(self):
        if False:
            return 10
        'Returns the classicalfunction arguments'
        if self._args is None:
            self.compile()
        return self._args

    @property
    def types(self):
        if False:
            print('Hello World!')
        'Dumps a list of scopes with their variables and types.\n\n        Returns:\n            list(dict): A list of scopes as dicts, where key is the variable name and\n            value is its type.\n        '
        ret = []
        for scope in self.scopes:
            ret.append({k: v[0] for (k, v) in scope.items()})
        return ret

    def simulate(self, bitstring: str) -> bool:
        if False:
            print('Hello World!')
        'Evaluate the expression on a bitstring.\n\n        This evaluation is done classically.\n\n        Args:\n            bitstring: The bitstring for which to evaluate.\n\n        Returns:\n            bool: result of the evaluation.\n        '
        from tweedledum.classical import simulate
        return simulate(self._network, bitstring)

    def simulate_all(self):
        if False:
            while True:
                i = 10
        '\n        Returns a truth table.\n\n        Returns:\n            str: a bitstring with a truth table\n        '
        result = []
        for position in range(2 ** self._network.num_pis()):
            sim_result = ''.join([str(int(tt[position])) for tt in self.truth_table])
            result.append(sim_result)
        return ''.join(reversed(result))

    @property
    def truth_table(self):
        if False:
            return 10
        'Returns (and computes) the truth table'
        from tweedledum.classical import simulate
        if self._truth_table is None:
            self._truth_table = simulate(self._network)
        return self._truth_table

    def synth(self, registerless: bool=True, synthesizer: Optional[Callable[[ClassicalElement], QuantumCircuit]]=None) -> QuantumCircuit:
        if False:
            for i in range(10):
                print('nop')
        "Synthesis the logic network into a :class:`~qiskit.circuit.QuantumCircuit`.\n\n        Args:\n            registerless: Default ``True``. If ``False`` uses the parameter names to create\n            registers with those names. Otherwise, creates a circuit with a flat quantum register.\n            synthesizer: Optional. If None tweedledum's pkrm_synth is used.\n\n        Returns:\n            QuantumCircuit: A circuit implementing the logic network.\n        "
        if registerless:
            qregs = None
        else:
            qregs = self.qregs
        if synthesizer:
            return synthesizer(self)
        from tweedledum.synthesis import pkrm_synth
        return tweedledum2qiskit(pkrm_synth(self.truth_table[0]), name=self.name, qregs=qregs)

    def _define(self):
        if False:
            print('Hello World!')
        'The definition of the classical function is its synthesis'
        self.definition = self.synth()

    @property
    def qregs(self):
        if False:
            for i in range(10):
                print('nop')
        'The list of qregs used by the classicalfunction'
        qregs = [QuantumRegister(1, name=arg) for arg in self.args if self.types[0][arg] == 'Int1']
        if self.types[0]['return'] == 'Int1':
            qregs.append(QuantumRegister(1, name='return'))
        return qregs