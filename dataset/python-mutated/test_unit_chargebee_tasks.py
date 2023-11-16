from organisations.chargebee.tasks import update_chargebee_cache

def test_update_chargebee_cache(mocker):
    if False:
        return 10
    mock_chargebee_cache = mocker.MagicMock()
    mocker.patch('organisations.chargebee.tasks.ChargebeeCache', return_value=mock_chargebee_cache)
    update_chargebee_cache()
    mock_chargebee_cache.refresh.assert_called_once_with()