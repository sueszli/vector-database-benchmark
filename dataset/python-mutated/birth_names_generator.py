from __future__ import annotations
from collections.abc import Iterable
from datetime import datetime
from random import choice, randint
from typing import Any, TYPE_CHECKING
from tests.consts.birth_names import BOY, DS, GENDER, GIRL, NAME, NUM, NUM_BOYS, NUM_GIRLS, STATE
from tests.consts.us_states import US_STATES
from tests.example_data.data_generator.base_generator import ExampleDataGenerator
if TYPE_CHECKING:
    from tests.example_data.data_generator.string_generator import StringGenerator

class BirthNamesGenerator(ExampleDataGenerator):
    _names_generator: StringGenerator
    _start_year: int
    _until_not_include_year: int
    _rows_per_year: int

    def __init__(self, names_generator: StringGenerator, start_year: int, years_amount: int, rows_per_year: int) -> None:
        if False:
            while True:
                i = 10
        assert start_year > -1
        assert years_amount > 0
        self._names_generator = names_generator
        self._start_year = start_year
        self._until_not_include_year = start_year + years_amount
        self._rows_per_year = rows_per_year

    def generate(self) -> Iterable[dict[Any, Any]]:
        if False:
            return 10
        for year in range(self._start_year, self._until_not_include_year):
            ds = self._make_year(year)
            for _ in range(self._rows_per_year):
                yield self.generate_row(ds)

    def _make_year(self, year: int):
        if False:
            i = 10
            return i + 15
        return datetime(year, 1, 1, 0, 0, 0)

    def generate_row(self, dt: datetime) -> dict[Any, Any]:
        if False:
            return 10
        gender = choice([BOY, GIRL])
        num = randint(1, 100000)
        return {DS: dt, GENDER: gender, NAME: self._names_generator.generate(), NUM: num, STATE: choice(US_STATES), NUM_BOYS: num if gender == BOY else 0, NUM_GIRLS: num if gender == GIRL else 0}