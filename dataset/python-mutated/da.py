"""Specific data provider for Denmark (da)."""
import operator
import typing as t
from mimesis import Datetime
from mimesis.locales import Locale
from mimesis.providers import BaseDataProvider
from mimesis.types import MissingSeed, Seed
__all__ = ['DenmarkSpecProvider']

class DenmarkSpecProvider(BaseDataProvider):
    """Class that provides special data for Denmark (da)."""

    def __init__(self, seed: Seed=MissingSeed) -> None:
        if False:
            while True:
                i = 10
        'Initialize attributes.'
        super().__init__(locale=Locale.DA, seed=seed)
        self._datetime = Datetime(locale=Locale.DA, seed=seed, random=self.random)
        self._checksum_factors = (4, 3, 2, 7, 6, 5, 4, 3, 2)

    class Meta:
        name = 'denmark_provider'
        datafile = None

    def _calculate_century_selector(self, year: int) -> int:
        if False:
            for i in range(10):
                print('nop')
        if 1858 <= year < 1900:
            return self.random.randint(5, 8)
        elif 1900 <= year < 1937:
            return self.random.randint(0, 3)
        elif 1937 <= year < 2000:
            return self.random.choice([4, 9])
        elif 2000 <= year < 2037:
            return self.random.randint(4, 9)
        raise ValueError('Invalid year')

    def _calculate_checksum(self, cpr_nr_no_checksum: str) -> int:
        if False:
            while True:
                i = 10
        'Calculate the CPR number checksum.\n\n        The CPR checksum can be checked by:\n        1. Multiplying each digit in the CPR number with a corresponding fixed\n           factor (self._checksum_factors) to produce a list of products.\n        2. Summing up all the products, including the checksum, and checking\n           that the resulting sum modulo 11 is 0.\n\n        As such the checksum can be determined by reordering the formula, as:\n        * 11 - (sum_without_checksum % 11)\n\n        If the sum_without_checksum is 0, the resulting checksum is 11, but\n        returned as 0 according to the official rules.\n\n        If the sum_without_checksum is 1, the resulting checksum is 10, and\n        thus invalid as the checksum is only 1 digit, hence this implies that\n        the generated serial_number is invalid.\n\n        Note: This method does not handle checksum == 10 case.\n              It is handled by recursion in _generate_serial_checksum.\n        '
        cpr_digits = map(int, cpr_nr_no_checksum)
        cpr_digit_products = list(map(operator.mul, cpr_digits, self._checksum_factors))
        remainder: int = sum(cpr_digit_products) % 11
        if remainder == 0:
            return 0
        return 11 - remainder

    def _generate_serial_checksum(self, cpr_century: str) -> t.Tuple[str, int]:
        if False:
            while True:
                i = 10
        'Generate a serial number and checksum from cpr_century.'
        serial_number = f'{self.random.randint(0, 99):02d}'
        cpr_nr_no_checksum = f'{cpr_century}{serial_number}'
        checksum = self._calculate_checksum(cpr_nr_no_checksum)
        if checksum == 10:
            return self._generate_serial_checksum(cpr_century)
        return (serial_number, checksum)

    def cpr(self) -> str:
        if False:
            print('Hello World!')
        'Generate a random CPR number (Central Person Registry).\n\n        :return: CPR number.\n\n        :Example:\n            0405420694\n        '
        date = self._datetime.date(start=1858, end=2021)
        cpr_date = f'{date:%d%m%y}'
        century_selector = self._calculate_century_selector(date.year)
        cpr_century = f'{cpr_date}{century_selector}'
        (serial_number, checksum) = self._generate_serial_checksum(cpr_century)
        cpr_nr = f'{cpr_century}{serial_number}{checksum}'
        return cpr_nr