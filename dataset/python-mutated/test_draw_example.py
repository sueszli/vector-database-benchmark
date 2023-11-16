import pytest
from hypothesis.strategies import lists
from tests.common import standard_types

@pytest.mark.parametrize('spec', standard_types, ids=list(map(repr, standard_types)))
def test_single_example(spec):
    if False:
        print('Hello World!')
    spec.example()

@pytest.mark.parametrize('spec', standard_types, ids=list(map(repr, standard_types)))
def test_list_example(spec):
    if False:
        return 10
    lists(spec).example()