"""
This module is responsible for generating the check digit and formatting
SBN numbers.
"""
from typing import Any, Optional

class SBN:
    MAX_LENGTH = 9

    def __init__(self, registrant: Optional[str]=None, publication: Optional[str]=None) -> None:
        if False:
            print('Hello World!')
        self.registrant = registrant
        self.publication = publication

class SBN9(SBN):

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__(*args, **kwargs)
        self.check_digit = self._check_digit()

    def _check_digit(self) -> str:
        if False:
            i = 10
            return i + 15
        'Calculate the check digit for SBN-9.\n        SBNs use the same check digit calculation as ISBN. See\n        https://en.wikipedia.org/wiki/International_Standard_Book_Number\n                for calculation. Only modification is weights range from 1 to 9\n                instead of 1 to 10.\n        '
        weights = range(1, 9)
        body = ''.join([part for part in [self.registrant, self.publication] if part is not None])
        remainder = sum((int(b) * w for (b, w) in zip(body, weights))) % 11
        check_digit = 'X' if remainder == 10 else str(remainder)
        return str(check_digit)

    def format(self, separator: str='') -> str:
        if False:
            return 10
        return separator.join([part for part in [self.registrant, self.publication, self.check_digit] if part is not None])