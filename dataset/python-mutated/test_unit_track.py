from unittest import mock
import pytest
from app_analytics.track import track_request_googleanalytics, track_request_influxdb

@pytest.mark.parametrize('request_uri, expected_ga_requests', (('/api/v1/flags/', 2), ('/api/v1/identities/', 2), ('/api/v1/traits/', 2), ('/api/v1/features/', 1), ('/health', 1)))
@mock.patch('app_analytics.track.requests')
@mock.patch('app_analytics.track.Environment')
def test_track_request_googleanalytics(MockEnvironment, mock_requests, request_uri, expected_ga_requests):
    if False:
        i = 10
        return i + 15
    "\n    Verify that the correct number of calls are made to GA for the various uris.\n\n    All SDK endpoints should send 2 requests as they send a page view and an event (for managing number of API\n    requests made by an organisation). All API requests made to the 'admin' API, for managing flags, etc. should\n    only send a page view request.\n    "
    request = mock.MagicMock()
    request.path = request_uri
    environment_api_key = 'test'
    request.headers = {'X-Environment-Key': environment_api_key}
    track_request_googleanalytics(request)
    assert mock_requests.post.call_count == expected_ga_requests

@pytest.mark.parametrize('request_uri, expected_resource', (('/api/v1/flags/', 'flags'), ('/api/v1/identities/', 'identities'), ('/api/v1/traits/', 'traits'), ('/api/v1/environment-document/', 'environment-document')))
@mock.patch('app_analytics.track.InfluxDBWrapper')
@mock.patch('app_analytics.track.Environment')
def test_track_request_sends_data_to_influxdb_for_tracked_uris(MockEnvironment, MockInfluxDBWrapper, request_uri, expected_resource):
    if False:
        for i in range(10):
            print('nop')
    '\n    Verify that the correct number of calls are made to InfluxDB for the various uris.\n    '
    request = mock.MagicMock()
    request.path = request_uri
    environment_api_key = 'test'
    request.headers = {'X-Environment-Key': environment_api_key}
    mock_influxdb = mock.MagicMock()
    MockInfluxDBWrapper.return_value = mock_influxdb
    track_request_influxdb(request)
    call_list = MockInfluxDBWrapper.call_args_list
    assert len(call_list) == 1
    assert mock_influxdb.add_data_point.call_args_list[0][1]['tags']['resource'] == expected_resource

@mock.patch('app_analytics.track.InfluxDBWrapper')
@mock.patch('app_analytics.track.Environment')
def test_track_request_sends_host_data_to_influxdb(MockEnvironment, MockInfluxDBWrapper, rf):
    if False:
        print('Hello World!')
    '\n    Verify that host is part of the data send to influxDB\n    '
    environment_api_key = 'test'
    headers = {'X-Environment-Key': environment_api_key}
    request = rf.get('/api/v1/flags/', headers=headers)
    mock_influxdb = mock.MagicMock()
    MockInfluxDBWrapper.return_value = mock_influxdb
    track_request_influxdb(request)
    assert mock_influxdb.add_data_point.call_args_list[0][1]['tags']['host'] == 'testserver'

@mock.patch('app_analytics.track.InfluxDBWrapper')
@mock.patch('app_analytics.track.Environment')
def test_track_request_does_not_send_data_to_influxdb_for_not_tracked_uris(MockEnvironment, MockInfluxDBWrapper):
    if False:
        for i in range(10):
            print('nop')
    '\n    Verify that the correct number of calls are made to InfluxDB for the various uris.\n    '
    request = mock.MagicMock()
    request.path = '/health'
    environment_api_key = 'test'
    request.headers = {'X-Environment-Key': environment_api_key}
    mock_influxdb = mock.MagicMock()
    MockInfluxDBWrapper.return_value = mock_influxdb
    track_request_influxdb(request)
    MockInfluxDBWrapper.assert_not_called()