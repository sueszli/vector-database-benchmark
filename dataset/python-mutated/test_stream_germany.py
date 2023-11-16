import pytest
from source_rki_covid.source import Germany

@pytest.fixture
def patch_germany_class(mocker):
    if False:
        i = 10
        return i + 15
    mocker.patch.object(Germany, 'primary_key', None)

def test_path(patch_germany_class):
    if False:
        print('Hello World!')
    stream = Germany()
    expected_params = {'path': 'germany/'}
    assert stream.path() == expected_params.get('path')