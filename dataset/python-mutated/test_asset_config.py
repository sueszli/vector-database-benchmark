import mock
from docs_snippets.concepts.assets.asset_config import MyDownstreamAssetConfig, my_downstream_asset

def test_my_configurable_asset() -> None:
    if False:
        i = 10
        return i + 15
    with mock.patch('requests.get') as mock_get:
        mock_get.return_value = mock.Mock()
        mock_get.return_value.json.return_value = {'value': 10}
        assert my_downstream_asset(MyDownstreamAssetConfig(api_endpoint='bar'), 5) == 15
        assert mock_get.call_args[0][0] == 'bar/data'
        assert my_downstream_asset(MyDownstreamAssetConfig(api_endpoint='bar'), 2) == 12