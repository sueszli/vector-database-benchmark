"""Specific data provider for the Netherlands (nl)."""
from mimesis.locales import Locale
from mimesis.providers import BaseDataProvider
from mimesis.types import MissingSeed, Seed
__all__ = ['NetherlandsSpecProvider']

class NetherlandsSpecProvider(BaseDataProvider):
    """Class that provides special data for the Netherlands (nl)."""

    def __init__(self, seed: Seed=MissingSeed) -> None:
        if False:
            return 10
        'Initialize attributes.'
        super().__init__(locale=Locale.NL, seed=seed)

    class Meta:
        name = 'netherlands_provider'
        datafile = None

    def bsn(self) -> str:
        if False:
            i = 10
            return i + 15
        'Generate a random, but valid ``Burgerservicenummer``.\n\n        :returns: Random BSN.\n\n        :Example:\n            255159705\n        '

        def _is_valid_bsn(number: str) -> bool:
            if False:
                while True:
                    i = 10
            total = 0
            multiplier = 9
            for char in number:
                multiplier = -multiplier if multiplier == 1 else multiplier
                total += int(char) * multiplier
                multiplier -= 1
            result = total % 11 == 0
            return result
        (a, b) = (100000000, 999999999)
        sample = str(self.random.randint(a, b))
        while not _is_valid_bsn(sample):
            sample = str(self.random.randint(a, b))
        return sample

    def burgerservicenummer(self) -> str:
        if False:
            print('Hello World!')
        'Generate a random, but valid ``Burgerservicenummer``.\n\n        An alias for self.bsn()\n        '
        return self.bsn()