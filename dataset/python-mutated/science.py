"""Provides pseudo-scientific data."""
import typing as t
from mimesis.data import SI_PREFIXES, SI_PREFIXES_SYM
from mimesis.enums import MeasureUnit, MetricPrefixSign
from mimesis.providers.base import BaseProvider
__all__ = ['Science']

class Science(BaseProvider):
    """Class for generating pseudo-scientific data."""

    class Meta:
        name = 'science'

    def rna_sequence(self, length: int=10) -> str:
        if False:
            while True:
                i = 10
        'Generate a random RNA sequence.\n\n        :param length: Length of block.\n        :return: RNA sequence.\n\n        :Example:\n            AGUGACACAA\n        '
        return self.random._generate_string('UCGA', length)

    def dna_sequence(self, length: int=10) -> str:
        if False:
            for i in range(10):
                print('nop')
        'Generate a random DNA sequence.\n\n        :param length: Length of block.\n        :return: DNA sequence.\n\n        :Example:\n            GCTTTAGACC\n        '
        return self.random._generate_string('TCGA', length)

    def measure_unit(self, name: t.Optional[MeasureUnit]=None, symbol: bool=False) -> str:
        if False:
            while True:
                i = 10
        'Get unit name from International System of Units.\n\n        :param name: Enum object UnitName.\n        :param symbol: Return only symbol\n        :return: Unit.\n        '
        result: t.Tuple[str, str] = self.validate_enum(item=name, enum=MeasureUnit)
        if symbol:
            return result[1]
        return result[0]

    def metric_prefix(self, sign: t.Optional[MetricPrefixSign]=None, symbol: bool=False) -> str:
        if False:
            for i in range(10):
                print('nop')
        'Get a random prefix for the International System of Units.\n\n        :param sign: Sing of prefix (positive/negative).\n        :param symbol: Return the symbol of the prefix.\n        :return: Metric prefix for SI measure units.\n        :raises NonEnumerableError: if sign is not supported.\n\n        :Example:\n            mega\n        '
        prefixes = SI_PREFIXES_SYM if symbol else SI_PREFIXES
        key = self.validate_enum(item=sign, enum=MetricPrefixSign)
        return self.random.choice(prefixes[key])