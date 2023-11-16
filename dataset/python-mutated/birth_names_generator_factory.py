from __future__ import annotations
from abc import ABC, abstractmethod
from tests.example_data.data_generator.birth_names.birth_names_generator import BirthNamesGenerator
from tests.example_data.data_generator.string_generator_factory import StringGeneratorFactory

class BirthNamesGeneratorFactory(ABC):
    __factory: BirthNamesGeneratorFactory

    @abstractmethod
    def _make(self) -> BirthNamesGenerator:
        if False:
            for i in range(10):
                print('nop')
        ...

    @classmethod
    def make(cls) -> BirthNamesGenerator:
        if False:
            while True:
                i = 10
        return cls._get_instance()._make()

    @classmethod
    def set_instance(cls, factory: BirthNamesGeneratorFactory) -> None:
        if False:
            return 10
        cls.__factory = factory

    @classmethod
    def _get_instance(cls) -> BirthNamesGeneratorFactory:
        if False:
            while True:
                i = 10
        if not hasattr(cls, '_BirthNamesGeneratorFactory__factory'):
            cls.__factory = BirthNamesGeneratorFactoryImpl()
        return cls.__factory
MIN_NAME_LEN = 3
MAX_NAME_SIZE = 10
START_YEAR = 1960
YEARS_AMOUNT = 60
ROW_PER_YEAR = 20

class BirthNamesGeneratorFactoryImpl(BirthNamesGeneratorFactory):

    def _make(self) -> BirthNamesGenerator:
        if False:
            for i in range(10):
                print('nop')
        string_generator = StringGeneratorFactory.make_lowercase_based(MIN_NAME_LEN, MAX_NAME_SIZE)
        return BirthNamesGenerator(string_generator, START_YEAR, YEARS_AMOUNT, ROW_PER_YEAR)