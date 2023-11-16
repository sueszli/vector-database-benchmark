"""Specific data provider for Russia (ru)."""
import typing as t
from datetime import datetime
from mimesis.enums import Gender
from mimesis.locales import Locale
from mimesis.providers import BaseDataProvider
from mimesis.types import MissingSeed, Seed
__all__ = ['RussiaSpecProvider']

class RussiaSpecProvider(BaseDataProvider):
    """Class that provides special data for Russia (ru)."""

    def __init__(self, seed: Seed=MissingSeed) -> None:
        if False:
            while True:
                i = 10
        'Initialize attributes.'
        super().__init__(locale=Locale.RU, seed=seed)
        self._current_year = str(datetime.now().year)

    class Meta:
        """The name of the provider."""
        name = 'russia_provider'
        datafile = 'builtin.json'

    def generate_sentence(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        'Generate sentence from the parts.\n\n        :return: Sentence.\n        '
        sentences = self.extract(['sentence'])
        sentence = [self.random.choice(sentences[k]) for k in ('head', 'p1', 'p2', 'tail')]
        return ' '.join(sentence)

    def patronymic(self, gender: t.Optional[Gender]=None) -> str:
        if False:
            while True:
                i = 10
        'Generate random patronymic name.\n\n        :param gender: Gender of person.\n        :return: Patronymic name.\n\n        :Example:\n            Алексеевна.\n        '
        gender = self.validate_enum(gender, Gender)
        patronymics: t.List[str] = self.extract(['patronymic', str(gender)])
        return self.random.choice(patronymics)

    def passport_series(self, year: t.Optional[int]=None) -> str:
        if False:
            i = 10
            return i + 15
        'Generate random series of passport.\n\n        :param year: Year of manufacture.\n        :type year: int or None\n        :return: Series.\n\n        :Example:\n            02 15.\n        '
        if not year:
            year = self.random.randint(10, int(self._current_year[2:]))
        region = self.random.randint(1, 99)
        return f'{region:02d} {year}'

    def passport_number(self) -> int:
        if False:
            print('Hello World!')
        'Generate random passport number.\n\n        :return: Number.\n\n        :Example:\n            560430\n        '
        return self.random.randint(100000, 999999)

    def series_and_number(self) -> str:
        if False:
            while True:
                i = 10
        'Generate a random passport number and series.\n\n        :return: Series and number.\n\n        :Example:\n            57 16 805199.\n        '
        series = self.passport_series()
        number = self.passport_number()
        return f'{series} {number}'

    def snils(self) -> str:
        if False:
            while True:
                i = 10
        'Generate snils with special algorithm.\n\n        :return: SNILS.\n\n        :Example:\n            41917492600.\n        '
        numbers = []
        control_codes = []
        for i in range(0, 9):
            numbers.append(self.random.randint(0, 9))
        for i in range(9, 0, -1):
            control_codes.append(numbers[9 - i] * i)
        control_code = sum(control_codes)
        code = ''.join(map(str, numbers))
        if control_code in (100, 101):
            _snils = code + '00'
            return _snils
        if control_code < 100:
            _snils = code + str(control_code)
            return _snils
        if control_code > 101:
            control_code = control_code % 101
            if control_code == 100:
                control_code = 0
            _snils = code + f'{control_code:02}'
            return _snils
        raise RuntimeError('Must not be reached')

    def inn(self) -> str:
        if False:
            return 10
        'Generate random, but valid ``INN``.\n\n        :return: INN.\n        '

        def control_sum(nums: t.List[int], t: str) -> int:
            if False:
                print('Hello World!')
            digits_dict = {'n2': [7, 2, 4, 10, 3, 5, 9, 4, 6, 8], 'n1': [3, 7, 2, 4, 10, 3, 5, 9, 4, 6, 8]}
            number = 0
            digits = digits_dict[t]
            for (i, _) in enumerate(digits, start=0):
                number += nums[i] * digits[i]
            return number % 11 % 10
        numbers = []
        for x in range(0, 10):
            numbers.append(self.random.randint(1 if x == 0 else 0, 9))
        n2 = control_sum(numbers, 'n2')
        numbers.append(n2)
        n1 = control_sum(numbers, 'n1')
        numbers.append(n1)
        return ''.join(map(str, numbers))

    def ogrn(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        'Generate random valid ``OGRN``.\n\n        :return: OGRN.\n\n        :Example:\n            4715113303725.\n        '
        numbers = []
        for _ in range(0, 12):
            numbers.append(self.random.randint(1 if _ == 0 else 0, 9))
        _ogrn = ''.join((str(i) for i in numbers))
        check_sum = str(int(_ogrn) % 11 % 10)
        return f'{_ogrn}{check_sum}'

    def bic(self) -> str:
        if False:
            while True:
                i = 10
        'Generate random ``BIC`` (Bank ID Code).\n\n        :return: BIC.\n\n        :Example:\n            044025575.\n        '
        country_code = '04'
        code = f'{self.random.randint(1, 10):02}'
        bank_number = f'{self.random.randint(0, 99):02}'
        bank_office = f'{self.random.randint(50, 999):03}'
        bic = country_code + code + bank_number + bank_office
        return bic

    def kpp(self) -> str:
        if False:
            while True:
                i = 10
        "Generate random ``KPP``.\n\n        :return: 'KPP'.\n\n        :Example:\n            560058652.\n        "
        tax_codes = ['7700', '7800', '5000', '0100', '0200', '0300', '0500', '0600', '0700', '0800', '0900', '1000', '1100', '1200', '1300', '1400', '1500', '1600', '1700', '1800', '1900', '2000', '2100', '2200', '2300', '2400', '2500', '2600', '2700', '2800', '2900', '3000', '3100', '3200', '3300', '3400', '3500', '3600', '3700', '3800', '3900', '4000', '4100', '4900', '5100', '5200', '5300', '5400', '5500', '5600', '5700', '5800', '5900', '6000', '6100', '6200', '6300', '6400', '6500', '6600', '6700', '6800', '6900', '7000', '7100', '7200', '7300', '7400', '7500', '7600', '7900', '8600', '8700', '8900', '9100', '9200', '9800', '9900', '9901', '9951', '9952', '9953', '9954', '9955', '9956', '9957', '9958', '9959', '9961', '9962', '9965', '9966', '9971', '9972', '9973', '9974', '9975', '9976', '9977', '9979', '9998']
        tax_code = tax_codes[self.random.randint(0, len(tax_codes) - 1)]
        reg_code = f'{self.random.randint(1, 99):02}'
        reg_number = f'{self.random.randint(1, 999):03}'
        kpp = tax_code + reg_code + reg_number
        return kpp