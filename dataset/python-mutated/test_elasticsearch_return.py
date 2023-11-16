"""
Test the elasticsearch returner
"""
import pytest
import salt.returners.elasticsearch_return as elasticsearch_return
from tests.support.mock import MagicMock, patch

@pytest.fixture
def configure_loader_modules():
    if False:
        for i in range(10):
            print('nop')
    return {elasticsearch_return: {}}

def test__virtual_no_elasticsearch():
    if False:
        while True:
            i = 10
    '\n    Test __virtual__ function when elasticsearch is not installed\n    and the elasticsearch module is not available\n    '
    result = elasticsearch_return.__virtual__()
    expected = (False, 'Elasticsearch module not availble.  Check that the elasticsearch library is installed.')
    assert expected == result

def test__virtual_with_elasticsearch():
    if False:
        i = 10
        return i + 15
    '\n    Test __virtual__ function when elasticsearch\n    and the elasticsearch module is not available\n    '
    with patch.dict(elasticsearch_return.__salt__, {'elasticsearch.index_exists': MagicMock()}):
        result = elasticsearch_return.__virtual__()
        expected = 'elasticsearch'
        assert expected == result