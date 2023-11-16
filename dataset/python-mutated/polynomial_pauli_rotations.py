"""Polynomially controlled Pauli-rotations."""
from __future__ import annotations
from itertools import product
from qiskit.circuit import QuantumRegister, QuantumCircuit
from qiskit.circuit.exceptions import CircuitError
from .functional_pauli_rotations import FunctionalPauliRotations

def _binomial_coefficients(n):
    if False:
        for i in range(10):
            print('nop')
    "Return a dictionary of binomial coefficients\n\n    Based-on/forked from sympy's binomial_coefficients() function [#]\n\n    .. [#] https://github.com/sympy/sympy/blob/sympy-1.5.1/sympy/ntheory/multinomial.py\n    "
    data = {(0, n): 1, (n, 0): 1}
    temp = 1
    for k in range(1, n // 2 + 1):
        temp = temp * (n - k + 1) // k
        data[k, n - k] = data[n - k, k] = temp
    return data

def _large_coefficients_iter(m, n):
    if False:
        return 10
    "Return an iterator of multinomial coefficients\n\n    Based-on/forked from sympy's multinomial_coefficients_iterator() function [#]\n\n    .. [#] https://github.com/sympy/sympy/blob/sympy-1.5.1/sympy/ntheory/multinomial.py\n    "
    if m < 2 * n or n == 1:
        coefficients = _multinomial_coefficients(m, n)
        for (key, value) in coefficients.items():
            yield (key, value)
    else:
        coefficients = _multinomial_coefficients(n, n)
        coefficients_dict = {}
        for (key, value) in coefficients.items():
            coefficients_dict[tuple(filter(None, key))] = value
        coefficients = coefficients_dict
        temp = [n] + [0] * (m - 1)
        temp_a = tuple(temp)
        b = tuple(filter(None, temp_a))
        yield (temp_a, coefficients[b])
        if n:
            j = 0
        else:
            j = m
        while j < m - 1:
            temp_j = temp[j]
            if j:
                temp[j] = 0
                temp[0] = temp_j
            if temp_j > 1:
                temp[j + 1] += 1
                j = 0
            else:
                j += 1
                temp[j] += 1
            temp[0] -= 1
            temp_a = tuple(temp)
            b = tuple(filter(None, temp_a))
            yield (temp_a, coefficients[b])

def _multinomial_coefficients(m, n):
    if False:
        i = 10
        return i + 15
    "Return an iterator of multinomial coefficients\n\n    Based-on/forked from sympy's multinomial_coefficients() function [#]\n\n    .. [#] https://github.com/sympy/sympy/blob/sympy-1.5.1/sympy/ntheory/multinomial.py\n    "
    if not m:
        if n:
            return {}
        return {(): 1}
    if m == 2:
        return _binomial_coefficients(n)
    if m >= 2 * n and n > 1:
        return dict(_large_coefficients_iter(m, n))
    if n:
        j = 0
    else:
        j = m
    temp = [n] + [0] * (m - 1)
    res = {tuple(temp): 1}
    while j < m - 1:
        temp_j = temp[j]
        if j:
            temp[j] = 0
            temp[0] = temp_j
        if temp_j > 1:
            temp[j + 1] += 1
            j = 0
            start = 1
            v = 0
        else:
            j += 1
            start = j + 1
            v = res[tuple(temp)]
            temp[j] += 1
        for k in range(start, m):
            if temp[k]:
                temp[k] -= 1
                v += res[tuple(temp)]
                temp[k] += 1
        temp[0] -= 1
        res[tuple(temp)] = v * temp_j // (n - temp[0])
    return res

class PolynomialPauliRotations(FunctionalPauliRotations):
    """A circuit implementing polynomial Pauli rotations.

    For a polynomial :math:`p(x)`, a basis state :math:`|i\\rangle` and a target qubit
    :math:`|0\\rangle` this operator acts as:

    .. math::

        |i\\rangle |0\\rangle \\mapsto \\cos\\left(\\frac{p(i)}{2}\\right) |i\\rangle |0\\rangle
        + \\sin\\left(\\frac{p(i)}{2}\\right) |i\\rangle |1\\rangle

    Let n be the number of qubits representing the state, d the degree of p(x) and q_i the qubits,
    where q_0 is the least significant qubit. Then for

    .. math::

        x = \\sum_{i=0}^{n-1} 2^i q_i,

    we can write

    .. math::

        p(x) = \\sum_{j=0}^{j=d} c_j x^j

    where :math:`c` are the input coefficients, ``coeffs``.
    """

    def __init__(self, num_state_qubits: int | None=None, coeffs: list[float] | None=None, basis: str='Y', name: str='poly') -> None:
        if False:
            return 10
        "Prepare an approximation to a state with amplitudes specified by a polynomial.\n\n        Args:\n            num_state_qubits: The number of qubits representing the state.\n            coeffs: The coefficients of the polynomial. ``coeffs[i]`` is the coefficient of the\n                i-th power of x. Defaults to linear: [0, 1].\n            basis: The type of Pauli rotation ('X', 'Y', 'Z').\n            name: The name of the circuit.\n        "
        self._coeffs = coeffs or [0, 1]
        super().__init__(num_state_qubits=num_state_qubits, basis=basis, name=name)

    @property
    def coeffs(self) -> list[float]:
        if False:
            for i in range(10):
                print('nop')
        'The coefficients of the polynomial.\n\n        ``coeffs[i]`` is the coefficient of the i-th power of the function input :math:`x`,\n        that means that the rotation angles are based on the coefficients value,\n        following the formula\n\n        .. math::\n\n            c_j x^j ,  j=0, ..., d\n\n        where :math:`d` is the degree of the polynomial :math:`p(x)` and :math:`c` are the coefficients\n        ``coeffs``.\n\n        Returns:\n            The coefficients of the polynomial.\n        '
        return self._coeffs

    @coeffs.setter
    def coeffs(self, coeffs: list[float]) -> None:
        if False:
            print('Hello World!')
        'Set the coefficients of the polynomial.\n\n        ``coeffs[i]`` is the coefficient of the i-th power of x.\n\n        Args:\n            The coefficients of the polynomial.\n        '
        self._invalidate()
        self._coeffs = coeffs

    @property
    def degree(self) -> int:
        if False:
            return 10
        'Return the degree of the polynomial, equals to the number of coefficients minus 1.\n\n        Returns:\n            The degree of the polynomial. If the coefficients have not been set, return 0.\n        '
        if self.coeffs:
            return len(self.coeffs) - 1
        return 0

    def _reset_registers(self, num_state_qubits):
        if False:
            return 10
        'Reset the registers.'
        if num_state_qubits is not None:
            qr_state = QuantumRegister(num_state_qubits, name='state')
            qr_target = QuantumRegister(1, name='target')
            self.qregs = [qr_state, qr_target]
        else:
            self.qregs = []

    def _check_configuration(self, raise_on_failure: bool=True) -> bool:
        if False:
            return 10
        'Check if the current configuration is valid.'
        valid = True
        if self.num_state_qubits is None:
            valid = False
            if raise_on_failure:
                raise AttributeError('The number of qubits has not been set.')
        if self.num_qubits < self.num_state_qubits + 1:
            valid = False
            if raise_on_failure:
                raise CircuitError('Not enough qubits in the circuit, need at least {}.'.format(self.num_state_qubits + 1))
        return valid

    def _get_rotation_coefficients(self) -> dict[tuple[int, ...], float]:
        if False:
            return 10
        'Compute the coefficient of each monomial.\n\n        Returns:\n            A dictionary with pairs ``{control_state: rotation angle}`` where ``control_state``\n            is a tuple of ``0`` or ``1`` bits.\n        '
        all_combinations = list(product([0, 1], repeat=self.num_state_qubits))
        valid_combinations = []
        for combination in all_combinations:
            if 0 < sum(combination) <= self.degree:
                valid_combinations += [combination]
        rotation_coeffs = {control_state: 0.0 for control_state in valid_combinations}
        for (i, coeff) in enumerate(self.coeffs[1:]):
            i += 1
            for (comb, num_combs) in _multinomial_coefficients(self.num_state_qubits, i).items():
                control_state: tuple[int, ...] = ()
                power = 1
                for (j, qubit) in enumerate(comb):
                    if qubit > 0:
                        control_state += (1,)
                        power *= 2 ** (j * qubit)
                    else:
                        control_state += (0,)
                rotation_coeffs[control_state] += coeff * num_combs * power
        return rotation_coeffs

    def _build(self):
        if False:
            for i in range(10):
                print('nop')
        'If not already built, build the circuit.'
        if self._is_built:
            return
        super()._build()
        circuit = QuantumCircuit(*self.qregs, name=self.name)
        qr_state = circuit.qubits[:self.num_state_qubits]
        qr_target = circuit.qubits[self.num_state_qubits]
        rotation_coeffs = self._get_rotation_coefficients()
        if self.basis == 'x':
            circuit.rx(self.coeffs[0], qr_target)
        elif self.basis == 'y':
            circuit.ry(self.coeffs[0], qr_target)
        else:
            circuit.rz(self.coeffs[0], qr_target)
        for c in rotation_coeffs:
            qr_control = []
            for (i, _) in enumerate(c):
                if c[i] > 0:
                    qr_control.append(qr_state[i])
            if len(qr_control) > 1:
                if self.basis == 'x':
                    circuit.mcrx(rotation_coeffs[c], qr_control, qr_target)
                elif self.basis == 'y':
                    circuit.mcry(rotation_coeffs[c], qr_control, qr_target)
                else:
                    circuit.mcrz(rotation_coeffs[c], qr_control, qr_target)
            elif len(qr_control) == 1:
                if self.basis == 'x':
                    circuit.crx(rotation_coeffs[c], qr_control[0], qr_target)
                elif self.basis == 'y':
                    circuit.cry(rotation_coeffs[c], qr_control[0], qr_target)
                else:
                    circuit.crz(rotation_coeffs[c], qr_control[0], qr_target)
        self.append(circuit.to_gate(), self.qubits)