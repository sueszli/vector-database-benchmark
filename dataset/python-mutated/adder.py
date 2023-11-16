"""Compute the sum of two equally sized qubit registers."""
from qiskit.circuit import QuantumCircuit

class Adder(QuantumCircuit):
    """Compute the sum of two equally sized qubit registers.

    For two registers :math:`|a\\rangle_n` and :math:|b\\rangle_n` with :math:`n` qubits each, an
    adder performs the following operation

    .. math::

        |a\\rangle_n |b\\rangle_n \\mapsto |a\\rangle_n |a + b\\rangle_{n + 1}.

    The quantum register :math:`|a\\rangle_n` (and analogously :math:`|b\\rangle_n`)

    .. math::

        |a\\rangle_n = |a_0\\rangle \\otimes \\cdots \\otimes |a_{n - 1}\\rangle,

    for :math:`a_i \\in \\{0, 1\\}`, is associated with the integer value

    .. math::

        a = 2^{0}a_{0} + 2^{1}a_{1} + \\cdots + 2^{n - 1}a_{n - 1}.

    """

    def __init__(self, num_state_qubits: int, name: str='Adder') -> None:
        if False:
            return 10
        '\n        Args:\n            num_state_qubits: The number of qubits in each of the registers.\n            name: The name of the circuit.\n        '
        super().__init__(name=name)
        self._num_state_qubits = num_state_qubits

    @property
    def num_state_qubits(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        'The number of state qubits, i.e. the number of bits in each input register.\n\n        Returns:\n            The number of state qubits.\n        '
        return self._num_state_qubits