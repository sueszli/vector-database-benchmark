from unittest import mock
import pytest
import composer_storage_trigger

@mock.patch('composer_storage_trigger.make_iap_request', side_effect=Exception('Bad request: JSON body error'))
def test_json_body_error(make_iap_request_mock):
    if False:
        return 10
    trigger_event = None
    with pytest.raises(Exception):
        composer_storage_trigger.trigger_dag(trigger_event)

@mock.patch('composer_storage_trigger.make_iap_request', side_effect=Exception('Error in IAP response: unauthorized'))
def test_iap_response_error(make_iap_request_mock):
    if False:
        while True:
            i = 10
    trigger_event = {'file': 'some-gcs-file'}
    with pytest.raises(Exception):
        composer_storage_trigger.trigger_dag(trigger_event)

@mock.patch('composer_storage_trigger.make_iap_request', autospec=True)
def test_experimental_api_endpoint(make_iap_request_mock):
    if False:
        print('Hello World!')
    composer_storage_trigger.trigger_dag({'test': 'a'})
    make_iap_request_mock.assert_called_once_with('https://YOUR-TENANT-PROJECT.appspot.com/api/experimental/dags/composer_sample_trigger_response_dag/dag_runs', 'YOUR-CLIENT-ID', method='POST', json={'conf': {'test': 'a'}, 'replace_microseconds': 'false'})

@mock.patch('composer_storage_trigger.make_iap_request', autospec=True)
@mock.patch('composer_storage_trigger.USE_EXPERIMENTAL_API', False)
def test_stable_api_endpoint(make_iap_request_mock):
    if False:
        while True:
            i = 10
    composer_storage_trigger.trigger_dag({'test': 'a'})
    make_iap_request_mock.assert_called_once_with('https://YOUR-TENANT-PROJECT.appspot.com/api/v1/dags/composer_sample_trigger_response_dag/dagRuns', 'YOUR-CLIENT-ID', method='POST', json={'conf': {'test': 'a'}})