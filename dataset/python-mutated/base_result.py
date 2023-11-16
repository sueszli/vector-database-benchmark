"""
Primitive result abstract base class
"""
from __future__ import annotations
from abc import ABC
from collections.abc import Iterator, Sequence
from dataclasses import fields
from typing import Any, Dict
from numpy import ndarray
ExperimentData = Dict[str, Any]

class BasePrimitiveResult(ABC):
    """Primitive result abstract base class.

    Base class for Primitive results meant to provide common functionality to all inheriting
    result dataclasses.
    """

    def __post_init__(self) -> None:
        if False:
            return 10
        '\n        Verify that all fields in any inheriting result dataclass are consistent, after\n        instantiation, with the number of experiments being represented.\n\n        This magic method is specific of `dataclasses.dataclass`, therefore all inheriting\n        classes must have this decorator.\n\n        Raises:\n            TypeError: If one of the data fields is not a Sequence or ``numpy.ndarray``.\n            ValueError: Inconsistent number of experiments across data fields.\n        '
        for value in self._field_values:
            if not isinstance(value, (Sequence, ndarray)) or isinstance(value, (str, bytes)):
                raise TypeError(f'Expected sequence or `numpy.ndarray`, provided {type(value)} instead.')
            if len(value) != self.num_experiments:
                raise ValueError('Inconsistent number of experiments across data fields.')

    @property
    def num_experiments(self) -> int:
        if False:
            i = 10
            return i + 15
        'Number of experiments in any inheriting result dataclass.'
        value: Sequence = self._field_values[0]
        return len(value)

    @property
    def experiments(self) -> tuple[ExperimentData, ...]:
        if False:
            return 10
        'Experiment data dicts in any inheriting result dataclass.'
        return tuple(self._generate_experiments())

    def _generate_experiments(self) -> Iterator[ExperimentData]:
        if False:
            print('Hello World!')
        'Generate experiment data dicts in any inheriting result dataclass.'
        names: tuple[str, ...] = self._field_names
        for values in zip(*self._field_values):
            yield dict(zip(names, values))

    def decompose(self) -> Iterator[BasePrimitiveResult]:
        if False:
            for i in range(10):
                print('nop')
        'Generate single experiment result objects from self.'
        for values in zip(*self._field_values):
            yield self.__class__(*[(v,) for v in values])

    @property
    def _field_names(self) -> tuple[str, ...]:
        if False:
            while True:
                i = 10
        'Tuple of field names in any inheriting result dataclass.'
        return tuple((field.name for field in fields(self)))

    @property
    def _field_values(self) -> tuple:
        if False:
            i = 10
            return i + 15
        'Tuple of field values in any inheriting result dataclass.'
        return tuple((getattr(self, name) for name in self._field_names))