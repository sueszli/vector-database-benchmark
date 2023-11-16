"""Primitive abstract base class."""
from __future__ import annotations
from abc import ABC
from collections.abc import Sequence
from qiskit.circuit import QuantumCircuit
from qiskit.providers import Options
from qiskit.utils.deprecation import deprecate_func
from . import validation

class BasePrimitive(ABC):
    """Primitive abstract base class."""

    def __init__(self, options: dict | None=None):
        if False:
            print('Hello World!')
        self._run_options = Options()
        if options is not None:
            self._run_options.update_options(**options)

    @property
    def options(self) -> Options:
        if False:
            for i in range(10):
                print('nop')
        'Return options values for the estimator.\n\n        Returns:\n            options\n        '
        return self._run_options

    def set_options(self, **fields):
        if False:
            while True:
                i = 10
        'Set options values for the estimator.\n\n        Args:\n            **fields: The fields to update the options\n        '
        self._run_options.update_options(**fields)

    @staticmethod
    @deprecate_func(since='0.46.0')
    def _validate_circuits(circuits: Sequence[QuantumCircuit] | QuantumCircuit) -> tuple[QuantumCircuit, ...]:
        if False:
            return 10
        return validation._validate_circuits(circuits)

    @staticmethod
    @deprecate_func(since='0.46.0')
    def _validate_parameter_values(parameter_values: Sequence[Sequence[float]] | Sequence[float] | float | None, default: Sequence[Sequence[float]] | Sequence[float] | None=None) -> tuple[tuple[float, ...], ...]:
        if False:
            i = 10
            return i + 15
        return validation._validate_parameter_values(parameter_values, default=default)

    @staticmethod
    @deprecate_func(since='0.46.0')
    def _cross_validate_circuits_parameter_values(circuits: tuple[QuantumCircuit, ...], parameter_values: tuple[tuple[float, ...], ...]) -> None:
        if False:
            return 10
        return validation._cross_validate_circuits_parameter_values(circuits, parameter_values=parameter_values)