import pytest
from tests.support.mock import Mock, call, patch
pytest.importorskip('mako')
from salt.utils.mako import SaltMakoTemplateLookup

def test_mako_template_lookup(minion_opts):
    if False:
        i = 10
        return i + 15
    '\n    The shudown method can be called without raising an exception when the\n    file_client does not have a destroy method\n    '
    file_client = Mock()
    with patch('salt.fileclient.get_file_client', return_value=file_client):
        loader = SaltMakoTemplateLookup(minion_opts)
        assert loader._file_client is None
        assert loader.file_client() is file_client
        assert loader._file_client is file_client
        try:
            loader.destroy()
        except AttributeError:
            pytest.fail('Regression when calling SaltMakoTemplateLookup.destroy()')
        assert file_client.mock_calls == [call.destroy()]