import os
from unittest import mock
import access_token
PROJECT = os.environ['GOOGLE_CLOUD_PROJECT']

@mock.patch('access_token.requests')
def test_main(requests_mock: mock.MagicMock) -> None:
    if False:
        for i in range(10):
            print('nop')
    metadata_response = mock.Mock()
    metadata_response.status_code = 200
    metadata_response.json.return_value = {'access_token': '123'}
    bucket_response = mock.Mock()
    bucket_response.status_code = 200
    bucket_response.json.return_value = [{'bucket': 'name'}]
    requests_mock.get.side_effect = [metadata_response, bucket_response]
    access_token.main(PROJECT)
    assert requests_mock.get.call_count == 2