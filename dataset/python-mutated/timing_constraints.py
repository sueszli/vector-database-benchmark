"""Timing Constraints class."""
from qiskit.transpiler.exceptions import TranspilerError

class TimingConstraints:
    """Hardware Instruction Timing Constraints."""

    def __init__(self, granularity: int=1, min_length: int=1, pulse_alignment: int=1, acquire_alignment: int=1):
        if False:
            for i in range(10):
                print('nop')
        'Initialize a TimingConstraints object\n\n        Args:\n            granularity: An integer value representing minimum pulse gate\n                resolution in units of ``dt``. A user-defined pulse gate should have\n                duration of a multiple of this granularity value.\n            min_length: An integer value representing minimum pulse gate\n                length in units of ``dt``. A user-defined pulse gate should be longer\n                than this length.\n            pulse_alignment: An integer value representing a time resolution of gate\n                instruction starting time. Gate instruction should start at time which\n                is a multiple of the alignment value.\n            acquire_alignment: An integer value representing a time resolution of measure\n                instruction starting time. Measure instruction should start at time which\n                is a multiple of the alignment value.\n\n        Notes:\n            This information will be provided by the backend configuration.\n\n        Raises:\n            TranspilerError: When any invalid constraint value is passed.\n        '
        self.granularity = granularity
        self.min_length = min_length
        self.pulse_alignment = pulse_alignment
        self.acquire_alignment = acquire_alignment
        for (key, value) in self.__dict__.items():
            if not isinstance(value, int) or value < 1:
                raise TranspilerError(f'Timing constraint {key} should be nonzero integer. Not {value}.')