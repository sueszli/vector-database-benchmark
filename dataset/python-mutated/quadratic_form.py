"""A circuit implementing a quadratic form on binary variables."""
from typing import Union, Optional, List
import numpy as np
from qiskit.circuit import QuantumCircuit, QuantumRegister, ParameterExpression
from ..basis_change import QFT

class QuadraticForm(QuantumCircuit):
    """Implements a quadratic form on binary variables encoded in qubit registers.

    A quadratic form on binary variables is a quadratic function :math:`Q` acting on a binary
    variable of :math:`n` bits, :math:`x = x_0 ... x_{n-1}`. For an integer matrix :math:`A`,
    an integer vector :math:`b` and an integer :math:`c` the function can be written as

    .. math::

        Q(x) = x^T A x + x^T b + c

    If :math:`A`, :math:`b` or :math:`c` contain scalar values, this circuit computes only
    an approximation of the quadratic form.

    Provided with :math:`m` qubits to encode the value, this circuit computes :math:`Q(x) \\mod 2^m`
    in [two's complement](https://stackoverflow.com/questions/1049722/what-is-2s-complement)
    representation.

    .. math::

        |x\\rangle_n |0\\rangle_m \\mapsto |x\\rangle_n |(Q(x) + 2^m) \\mod 2^m \\rangle_m

    Since we use two's complement e.g. the value of :math:`Q(x) = 3` requires 2 bits to represent
    the value and 1 bit for the sign: `3 = '011'` where the first `0` indicates a positive value.
    On the other hand, :math:`Q(x) = -3` would be `-3 = '101'`, where the first `1` indicates
    a negative value and `01` is the two's complement of `3`.

    If the value of :math:`Q(x)` is too large to be represented with `m` qubits, the resulting
    bitstring is :math:`(Q(x) + 2^m) \\mod 2^m)`.

    The implementation of this circuit is discussed in [1], Fig. 6.

    References:
        [1]: Gilliam et al., Grover Adaptive Search for Constrained Polynomial Binary Optimization.
             `arXiv:1912.04088 <https://arxiv.org/pdf/1912.04088.pdf>`_

    """

    def __init__(self, num_result_qubits: Optional[int]=None, quadratic: Optional[Union[np.ndarray, List[List[Union[float, ParameterExpression]]]]]=None, linear: Optional[Union[np.ndarray, List[Union[float, ParameterExpression]]]]=None, offset: Optional[Union[float, ParameterExpression]]=None, little_endian: bool=True) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Args:\n            num_result_qubits: The number of qubits to encode the result. Called :math:`m` in\n                the class documentation.\n            quadratic: A matrix containing the quadratic coefficients, :math:`A`.\n            linear: An array containing the linear coefficients, :math:`b`.\n            offset: A constant offset, :math:`c`.\n            little_endian: Encode the result in little endianness.\n\n        Raises:\n            ValueError: If ``linear`` and ``quadratic`` have mismatching sizes.\n            ValueError: If ``num_result_qubits`` is unspecified but cannot be determined because\n                some values of the quadratic form are parameterized.\n        '
        if quadratic is not None and linear is not None:
            if len(quadratic) != len(linear):
                raise ValueError('Mismatching sizes of quadratic and linear.')
        if quadratic is None:
            quadratic = []
        if linear is None:
            linear = []
        if offset is None:
            offset = 0
        num_input_qubits = np.max([1, len(linear), len(quadratic)])
        if num_result_qubits is None:
            if any((any((isinstance(q_ij, ParameterExpression) for q_ij in q_i)) for q_i in quadratic)) or any((isinstance(l_i, ParameterExpression) for l_i in linear)) or isinstance(offset, ParameterExpression):
                raise ValueError('If the number of result qubits is not specified, the quadratic form matrices/vectors/offset may not be parameterized.')
            num_result_qubits = self.required_result_qubits(quadratic, linear, offset)
        qr_input = QuantumRegister(num_input_qubits)
        qr_result = QuantumRegister(num_result_qubits)
        circuit = QuantumCircuit(qr_input, qr_result, name='Q(x)')
        if len(quadratic) == 0:
            quadratic = None
        if len(linear) == 0:
            linear = None
        scaling = np.pi * 2 ** (1 - num_result_qubits)
        circuit.h(qr_result)
        if little_endian:
            qr_result = qr_result[::-1]
        if offset != 0:
            for (i, q_i) in enumerate(qr_result):
                circuit.p(scaling * 2 ** i * offset, q_i)
        for j in range(num_input_qubits):
            value = linear[j] if linear is not None else 0
            value += quadratic[j][j] if quadratic is not None else 0
            if value != 0:
                for (i, q_i) in enumerate(qr_result):
                    circuit.cp(scaling * 2 ** i * value, qr_input[j], q_i)
        if quadratic is not None:
            for j in range(num_input_qubits):
                for k in range(j + 1, num_input_qubits):
                    value = quadratic[j][k] + quadratic[k][j]
                    if value != 0:
                        for (i, q_i) in enumerate(qr_result):
                            circuit.mcp(scaling * 2 ** i * value, [qr_input[j], qr_input[k]], q_i)
        iqft = QFT(num_result_qubits, do_swaps=False).inverse().reverse_bits()
        circuit.compose(iqft, qubits=qr_result[:], inplace=True)
        super().__init__(*circuit.qregs, name='Q(x)')
        self.compose(circuit.to_gate(), qubits=self.qubits, inplace=True)

    @staticmethod
    def required_result_qubits(quadratic: Union[np.ndarray, List[List[float]]], linear: Union[np.ndarray, List[float]], offset: float) -> int:
        if False:
            while True:
                i = 10
        'Get the number of required result qubits.\n\n        Args:\n            quadratic: A matrix containing the quadratic coefficients.\n            linear: An array containing the linear coefficients.\n            offset: A constant offset.\n\n        Returns:\n            The number of qubits needed to represent the value of the quadratic form\n            in twos complement.\n        '
        bounds = []
        for condition in [lambda x: x < 0, lambda x: x > 0]:
            bound = 0.0
            bound += sum((sum((q_ij for q_ij in q_i if condition(q_ij))) for q_i in quadratic))
            bound += sum((l_i for l_i in linear if condition(l_i)))
            bound += offset if condition(offset) else 0
            bounds.append(bound)
        num_qubits_for_min = int(np.ceil(np.log2(max(-bounds[0], 1))))
        num_qubits_for_max = int(np.ceil(np.log2(bounds[1] + 1)))
        num_result_qubits = 1 + max(num_qubits_for_min, num_qubits_for_max)
        return num_result_qubits