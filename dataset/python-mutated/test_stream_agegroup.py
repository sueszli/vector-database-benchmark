import pytest
from source_rki_covid.source import GermanyAgeGroups

@pytest.fixture
def patch_age_group(mocker):
    if False:
        i = 10
        return i + 15
    mocker.patch.object(GermanyAgeGroups, 'primary_key', None)

def test_path(patch_age_group):
    if False:
        return 10
    stream = GermanyAgeGroups()
    expected_params = {'path': 'germany/age-groups'}
    assert stream.path() == expected_params.get('path')