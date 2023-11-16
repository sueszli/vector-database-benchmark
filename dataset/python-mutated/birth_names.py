from __future__ import annotations
from typing import Callable, TYPE_CHECKING
from pytest import fixture
from tests.example_data.data_generator.birth_names.birth_names_generator_factory import BirthNamesGeneratorFactory
from tests.example_data.data_loading.data_definitions.birth_names import BirthNamesMetaDataFactory
if TYPE_CHECKING:
    from tests.example_data.data_generator.birth_names.birth_names_generator import BirthNamesGenerator
    from tests.example_data.data_loading.data_definitions.types import Table

@fixture(scope='session')
def birth_names_data_generator() -> BirthNamesGenerator:
    if False:
        while True:
            i = 10
    return BirthNamesGeneratorFactory.make()

@fixture(scope='session')
def birth_names_table_factory(birth_names_data_generator: BirthNamesGenerator, support_datetime_type: bool) -> Callable[[], Table]:
    if False:
        i = 10
        return i + 15

    def _birth_names_table_factory() -> Table:
        if False:
            i = 10
            return i + 15
        return BirthNamesMetaDataFactory(support_datetime_type).make_table(data=birth_names_data_generator.generate())
    return _birth_names_table_factory