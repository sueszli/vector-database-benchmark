"""Provides data related to food."""
import typing as t
from mimesis.providers.base import BaseDataProvider
__all__ = ['Food']

class Food(BaseDataProvider):
    """Class for generating data related to food."""

    class Meta:
        name = 'food'
        datafile = f'{name}.json'

    def _choice_from(self, key: str) -> str:
        if False:
            return 10
        'Choice random element.'
        data: t.List[str] = self.extract([key])
        return self.random.choice(data)

    def vegetable(self) -> str:
        if False:
            while True:
                i = 10
        'Get a random vegetable.\n\n        :return: Vegetable name.\n\n        :Example:\n            Tomato.\n        '
        return self._choice_from('vegetables')

    def fruit(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        'Get a random fruit or berry.\n\n        :return: Fruit name.\n\n        :Example:\n            Banana.\n        '
        return self._choice_from('fruits')

    def dish(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        'Get a random dish.\n\n        :return: Dish name.\n\n        :Example:\n            Ratatouille.\n        '
        return self._choice_from('dishes')

    def spices(self) -> str:
        if False:
            while True:
                i = 10
        'Get a random spices or herbs.\n\n        :return: Spices or herbs.\n\n        :Example:\n            Anise.\n        '
        return self._choice_from('spices')

    def drink(self) -> str:
        if False:
            while True:
                i = 10
        'Get a random drink.\n\n        :return: Alcoholic drink.\n\n        :Example:\n            Vodka.\n        '
        return self._choice_from('drinks')