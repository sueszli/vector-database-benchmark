"""
This module specifies how to configure the Python version in Pyre.
"""
import dataclasses
from . import exceptions

@dataclasses.dataclass(frozen=True)
class PythonVersion:
    major: int
    minor: int = 0
    micro: int = 0

    @staticmethod
    def from_string(input: str) -> 'PythonVersion':
        if False:
            while True:
                i = 10
        try:
            splits = input.split('.')
            if len(splits) == 1:
                return PythonVersion(major=int(splits[0]))
            elif len(splits) == 2:
                return PythonVersion(major=int(splits[0]), minor=int(splits[1]))
            elif len(splits) == 3:
                return PythonVersion(major=int(splits[0]), minor=int(splits[1]), micro=int(splits[2]))
            raise exceptions.InvalidPythonVersion("Version string is expected to have the form of 'X.Y.Z' but got " + f"'{input}'")
        except ValueError as error:
            raise exceptions.InvalidPythonVersion(str(error))

    def to_string(self) -> str:
        if False:
            while True:
                i = 10
        return f'{self.major}.{self.minor}.{self.micro}'