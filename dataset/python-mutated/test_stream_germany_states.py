import pytest
from source_rki_covid.source import GermanyStates

@pytest.fixture
def patch_germany_states_class(mocker):
    if False:
        return 10
    mocker.patch.object(GermanyStates, 'primary_key', None)

def test_path(patch_germany_states_class):
    if False:
        while True:
            i = 10
    stream = GermanyStates()
    expected_params = {'path': 'states/'}
    assert stream.path() == expected_params.get('path')