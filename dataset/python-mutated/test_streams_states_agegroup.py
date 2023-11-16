import pytest
from source_rki_covid.source import GermanyStatesAgeGroups

@pytest.fixture
def patch_states_age_group(mocker):
    if False:
        while True:
            i = 10
    mocker.patch.object(GermanyStatesAgeGroups, 'primary_key', None)

def test_path(patch_states_age_group):
    if False:
        return 10
    stream = GermanyStatesAgeGroups()
    expected_params = {'path': 'states/age-groups'}
    assert stream.path() == expected_params.get('path')