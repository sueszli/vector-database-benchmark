"""Management of schedule block references."""
from typing import Tuple
from collections import UserDict
from qiskit.pulse.exceptions import PulseError

class ReferenceManager(UserDict):
    """Dictionary wrapper to manage pulse schedule references."""

    def unassigned(self) -> Tuple[Tuple[str, ...], ...]:
        if False:
            i = 10
            return i + 15
        'Get the keys of unassigned references.\n\n        Returns:\n            Tuple of reference keys.\n        '
        keys = []
        for (key, value) in self.items():
            if value is None:
                keys.append(key)
        return tuple(keys)

    def __setitem__(self, key, value):
        if False:
            i = 10
            return i + 15
        if key in self and self[key] is not None:
            if self[key] != value:
                raise PulseError(f'Subroutine {key} is already assigned to the reference of the current scope, however, the newly assigned schedule conflicts with the existing schedule. This operation was not successfully done.')
            return
        super().__setitem__(key, value)

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        keys = ', '.join(map(repr, self.keys()))
        return f'{self.__class__.__name__}(references=[{keys}])'

    def __str__(self):
        if False:
            print('Hello World!')
        out = f'{self.__class__.__name__}:'
        for (key, reference) in self.items():
            prog_repr = repr(reference)
            if len(prog_repr) > 50:
                prog_repr = prog_repr[:50] + '...'
            out += f'\n  - {repr(key)}: {prog_repr}'
        return out