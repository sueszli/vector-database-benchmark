"""Specific data provider for Ukraine (uk)."""
import typing as t
from mimesis.enums import Gender
from mimesis.locales import Locale
from mimesis.providers import BaseDataProvider
from mimesis.types import MissingSeed, Seed
__all__ = ['UkraineSpecProvider']

class UkraineSpecProvider(BaseDataProvider):
    """Class that provides special data for Ukraine (uk)."""

    def __init__(self, seed: Seed=MissingSeed) -> None:
        if False:
            return 10
        'Initialize attributes.'
        super().__init__(locale=Locale.UK, seed=seed)

    class Meta:
        name = 'ukraine_provider'
        datafile = 'builtin.json'

    def patronymic(self, gender: t.Optional[Gender]=None) -> str:
        if False:
            print('Hello World!')
        'Generate random patronymic name.\n\n        :param gender: Gender of person.\n        :type gender: str or int\n        :return: Patronymic name.\n        '
        gender = self.validate_enum(gender, Gender)
        patronymics: t.List[str] = self.extract(['patronymic', str(gender)])
        return self.random.choice(patronymics)