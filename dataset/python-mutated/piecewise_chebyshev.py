"""Piecewise polynomial Chebyshev approximation to a given f(x)."""
from __future__ import annotations
from typing import Callable
import numpy as np
from numpy.polynomial.chebyshev import Chebyshev
from qiskit.circuit import QuantumRegister, AncillaRegister
from qiskit.circuit.library.blueprintcircuit import BlueprintCircuit
from qiskit.circuit.exceptions import CircuitError
from .piecewise_polynomial_pauli_rotations import PiecewisePolynomialPauliRotations

class PiecewiseChebyshev(BlueprintCircuit):
    """Piecewise Chebyshev approximation to an input function.

    For a given function :math:`f(x)` and degree :math:`d`, this class implements a piecewise
    polynomial Chebyshev approximation on :math:`n` qubits to :math:`f(x)` on the given intervals.
    All the polynomials in the approximation are of degree :math:`d`.

    The values of the parameters are calculated according to [1] and see [2] for a more
    detailed explanation of the circuit construction and how it acts on the qubits.

    Examples:

        .. plot::
           :include-source:

            import numpy as np
            from qiskit import QuantumCircuit
            from qiskit.circuit.library.arithmetic.piecewise_chebyshev import PiecewiseChebyshev
            f_x, degree, breakpoints, num_state_qubits = lambda x: np.arcsin(1 / x), 2, [2, 4], 2
            pw_approximation = PiecewiseChebyshev(f_x, degree, breakpoints, num_state_qubits)
            pw_approximation._build()
            qc = QuantumCircuit(pw_approximation.num_qubits)
            qc.h(list(range(num_state_qubits)))
            qc.append(pw_approximation.to_instruction(), qc.qubits)
            qc.draw(output='mpl')

    References:

        [1]: Haener, T., Roetteler, M., & Svore, K. M. (2018).
             Optimizing Quantum Circuits for Arithmetic.
             `arXiv:1805.12445 <http://arxiv.org/abs/1805.12445>`_
        [2]: Carrera Vazquez, A., Hiptmair, H., & Woerner, S. (2022).
             Enhancing the Quantum Linear Systems Algorithm Using Richardson Extrapolation.
             `ACM Transactions on Quantum Computing 3, 1, Article 2 <https://doi.org/10.1145/3490631>`_
    """

    def __init__(self, f_x: float | Callable[[int], float], degree: int | None=None, breakpoints: list[int] | None=None, num_state_qubits: int | None=None, name: str='pw_cheb') -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Args:\n            f_x: the function to be approximated. Constant functions should be specified\n             as f_x = constant.\n            degree: the degree of the polynomials.\n                Defaults to ``1``.\n            breakpoints: the breakpoints to define the piecewise-linear function.\n                Defaults to the full interval.\n            num_state_qubits: number of qubits representing the state.\n            name: The name of the circuit object.\n        '
        super().__init__(name=name)
        self._num_state_qubits = None
        self._f_x = f_x
        self._degree = degree if degree is not None else 1
        self._breakpoints = breakpoints if breakpoints is not None else [0]
        self._polynomials: list[list[float]] | None = None
        self.num_state_qubits = num_state_qubits

    def _check_configuration(self, raise_on_failure: bool=True) -> bool:
        if False:
            while True:
                i = 10
        'Check if the current configuration is valid.'
        valid = True
        if self._f_x is None:
            valid = False
            if raise_on_failure:
                raise AttributeError('The function to be approximated has not been set.')
        if self._degree is None:
            valid = False
            if raise_on_failure:
                raise AttributeError('The degree of the polynomials has not been set.')
        if self._breakpoints is None:
            valid = False
            if raise_on_failure:
                raise AttributeError('The breakpoints have not been set.')
        if self.num_state_qubits is None:
            valid = False
            if raise_on_failure:
                raise AttributeError('The number of qubits has not been set.')
        if self.num_qubits < self.num_state_qubits + 1:
            valid = False
            if raise_on_failure:
                raise CircuitError('Not enough qubits in the circuit, need at least {}.'.format(self.num_state_qubits + 1))
        return valid

    @property
    def f_x(self) -> float | Callable[[int], float]:
        if False:
            print('Hello World!')
        'The function to be approximated.\n\n        Returns:\n            The function to be approximated.\n        '
        return self._f_x

    @f_x.setter
    def f_x(self, f_x: float | Callable[[int], float] | None) -> None:
        if False:
            print('Hello World!')
        'Set the function to be approximated.\n\n        Note that this may change the underlying quantum register, if the number of state qubits\n        changes.\n\n        Args:\n            f_x: The new function to be approximated.\n        '
        if self._f_x is None or f_x != self._f_x:
            self._invalidate()
            self._f_x = f_x
            self._reset_registers(self.num_state_qubits)

    @property
    def degree(self) -> int:
        if False:
            return 10
        'The degree of the polynomials.\n\n        Returns:\n            The degree of the polynomials.\n        '
        return self._degree

    @degree.setter
    def degree(self, degree: int | None) -> None:
        if False:
            print('Hello World!')
        'Set the error tolerance.\n\n        Note that this may change the underlying quantum register, if the number of state qubits\n        changes.\n\n        Args:\n            degree: The new degree.\n        '
        if self._degree is None or degree != self._degree:
            self._invalidate()
            self._degree = degree
            self._reset_registers(self.num_state_qubits)

    @property
    def breakpoints(self) -> list[int]:
        if False:
            for i in range(10):
                print('nop')
        'The breakpoints for the piecewise approximation.\n\n        Returns:\n            The breakpoints for the piecewise approximation.\n        '
        breakpoints = self._breakpoints
        if self.num_state_qubits is not None:
            num_states = 2 ** self.num_state_qubits
            if breakpoints[-1] < num_states:
                breakpoints = breakpoints + [num_states]
            if breakpoints[0] > 0:
                breakpoints = [0] + breakpoints
        return breakpoints

    @breakpoints.setter
    def breakpoints(self, breakpoints: list[int] | None) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Set the breakpoints for the piecewise approximation.\n\n        Note that this may change the underlying quantum register, if the number of state qubits\n        changes.\n\n        Args:\n            breakpoints: The new breakpoints for the piecewise approximation.\n        '
        if self._breakpoints is None or breakpoints != self._breakpoints:
            self._invalidate()
            self._breakpoints = breakpoints if breakpoints is not None else [0]
            self._reset_registers(self.num_state_qubits)

    @property
    def polynomials(self) -> list[list[float]]:
        if False:
            i = 10
            return i + 15
        'The polynomials for the piecewise approximation.\n\n        Returns:\n            The polynomials for the piecewise approximation.\n\n        Raises:\n            TypeError: If the input function is not in the correct format.\n        '
        if self.num_state_qubits is None:
            return [[]]
        breakpoints = self._breakpoints
        if breakpoints == [0]:
            breakpoints = [0, 2 ** self.num_state_qubits]
        num_intervals = len(breakpoints)
        polynomials = []
        for i in range(0, num_intervals - 1):
            try:
                if isinstance(self.f_x, (float, int)):
                    polynomials.append([self.f_x])
                else:
                    poly = Chebyshev.interpolate(self.f_x, self.degree, domain=[breakpoints[i], breakpoints[i + 1]])
                    poly = 2 * poly.convert(kind=np.polynomial.Polynomial).coef
                    polynomials.append(poly.tolist())
            except ValueError as err:
                raise TypeError(" <lambda>() missing 1 required positional argument: '" + self.f_x.__code__.co_varnames[0] + "'." + " Constant functions should be specified as 'f_x = constant'.") from err
        if breakpoints[-1] < 2 ** self.num_state_qubits:
            polynomials = polynomials + [[2 * np.arcsin(1)]]
        if breakpoints[0] > 0:
            polynomials = [[2 * np.arcsin(1)]] + polynomials
        return polynomials

    @polynomials.setter
    def polynomials(self, polynomials: list[list[float]] | None) -> None:
        if False:
            print('Hello World!')
        'Set the polynomials for the piecewise approximation.\n\n        Note that this may change the underlying quantum register, if the number of state qubits\n        changes.\n\n        Args:\n            polynomials: The new breakpoints for the piecewise approximation.\n        '
        if self._polynomials is None or polynomials != self._polynomials:
            self._invalidate()
            self._polynomials = polynomials
            self._reset_registers(self.num_state_qubits)

    @property
    def num_state_qubits(self) -> int:
        if False:
            return 10
        'The number of state qubits representing the state :math:`|x\\rangle`.\n\n        Returns:\n            The number of state qubits.\n        '
        return self._num_state_qubits

    @num_state_qubits.setter
    def num_state_qubits(self, num_state_qubits: int | None) -> None:
        if False:
            return 10
        'Set the number of state qubits.\n\n        Note that this may change the underlying quantum register, if the number of state qubits\n        changes.\n\n        Args:\n            num_state_qubits: The new number of qubits.\n        '
        if self._num_state_qubits is None or num_state_qubits != self._num_state_qubits:
            self._invalidate()
            self._num_state_qubits = num_state_qubits
            if num_state_qubits is not None and self._breakpoints is None:
                self.breakpoints = [0, 2 ** num_state_qubits]
            self._reset_registers(num_state_qubits)

    def _reset_registers(self, num_state_qubits: int | None) -> None:
        if False:
            while True:
                i = 10
        'Reset the registers.'
        self.qregs = []
        if num_state_qubits is not None:
            qr_state = QuantumRegister(num_state_qubits, 'state')
            qr_target = QuantumRegister(1, 'target')
            self.qregs = [qr_state, qr_target]
            num_ancillas = num_state_qubits
            if num_ancillas > 0:
                qr_ancilla = AncillaRegister(num_ancillas)
                self.add_register(qr_ancilla)

    def _build(self):
        if False:
            i = 10
            return i + 15
        'Build the circuit if not already build. The operation is considered successful\n        when q_objective is :math:`|1>`'
        if self._is_built:
            return
        super()._build()
        poly_r = PiecewisePolynomialPauliRotations(self.num_state_qubits, self.breakpoints, self.polynomials, name=self.name)
        self.append(poly_r.to_gate(), self.qubits)