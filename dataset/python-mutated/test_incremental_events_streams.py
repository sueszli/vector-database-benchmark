from copy import deepcopy
from unittest.mock import DEFAULT, MagicMock, Mock, call
import pendulum
import pytest
from airbyte_cdk.models import SyncMode
from airbyte_cdk.utils import AirbyteTracedException
from source_google_ads.google_ads import GoogleAds
from source_google_ads.streams import CampaignCriterion, ChangeStatus

def mock_response_parent():
    if False:
        while True:
            i = 10
    yield [{'change_status.last_change_date_time': '2023-06-13 12:36:01.772447', 'change_status.resource_type': 'CAMPAIGN_CRITERION', 'change_status.resource_status': 'ADDED', 'change_status.campaign_criterion': '1'}, {'change_status.last_change_date_time': '2023-06-13 12:36:02.772447', 'change_status.resource_type': 'CAMPAIGN_CRITERION', 'change_status.resource_status': 'ADDED', 'change_status.campaign_criterion': '2'}, {'change_status.last_change_date_time': '2023-06-13 12:36:03.772447', 'change_status.resource_type': 'CAMPAIGN_CRITERION', 'change_status.resource_status': 'REMOVED', 'change_status.campaign_criterion': '3'}, {'change_status.last_change_date_time': '2023-06-13 12:36:04.772447', 'change_status.resource_type': 'CAMPAIGN_CRITERION', 'change_status.resource_status': 'REMOVED', 'change_status.campaign_criterion': '4'}]

def mock_response_child():
    if False:
        print('Hello World!')
    yield [{'customer.id': 123, 'campaign.id': 1, 'campaign_criterion.resource_name': '1'}, {'customer.id': 123, 'campaign.id': 1, 'campaign_criterion.resource_name': '2'}]

class MockGoogleAds(GoogleAds):

    def parse_single_result(self, schema, result):
        if False:
            while True:
                i = 10
        return result

    def send_request(self, query: str, customer_id: str):
        if False:
            print('Hello World!')
        if query == 'query_parent':
            return mock_response_parent()
        else:
            return mock_response_child()

def test_change_status_stream(config, customers):
    if False:
        return 10
    ' '
    customer_id = next(iter(customers)).id
    stream_slice = {'customer_id': customer_id}
    google_api = MockGoogleAds(credentials=config['credentials'])
    stream = ChangeStatus(api=google_api, customers=customers)
    stream.get_query = Mock()
    stream.get_query.return_value = 'query_parent'
    result = list(stream.read_records(sync_mode=SyncMode.incremental, cursor_field=['change_status.last_change_date_time'], stream_slice=stream_slice))
    assert len(result) == 4
    assert stream.get_query.call_count == 1
    stream.get_query.assert_called_with({'customer_id': customer_id})

def test_child_incremental_events_read(config, customers):
    if False:
        for i in range(10):
            print('nop')
    '\n    Page token expired while reading records on date 2021-01-03\n    The latest read record is {"segments.date": "2021-01-03", "click_view.gclid": "4"}\n    It should retry reading starting from 2021-01-03, already read records will be reread again from that date.\n    It shouldn\'t read records on 2021-01-01, 2021-01-02\n    '
    customer_id = next(iter(customers)).id
    parent_stream_slice = {'customer_id': customer_id, 'resource_type': 'CAMPAIGN_CRITERION'}
    stream_state = {'change_status': {customer_id: {'change_status.last_change_date_time': '2023-08-16 13:20:01.003295'}}}
    google_api = MockGoogleAds(credentials=config['credentials'])
    stream = CampaignCriterion(api=google_api, customers=customers)
    parent_stream = stream.parent_stream
    parent_stream.get_query = Mock()
    parent_stream.get_query.return_value = 'query_parent'
    parent_stream.stream_slices = Mock()
    parent_stream.stream_slices.return_value = [parent_stream_slice]
    parent_stream.state = {customer_id: {'change_status.last_change_date_time': '2023-05-16 13:20:01.003295'}}
    stream.get_query = Mock()
    stream.get_query.return_value = 'query_child'
    stream_slices = list(stream.stream_slices(stream_state=stream_state))
    assert stream_slices == [{'customer_id': '123', 'updated_ids': {'2', '1'}, 'deleted_ids': {'3', '4'}, 'record_changed_time_map': {'1': '2023-06-13 12:36:01.772447', '2': '2023-06-13 12:36:02.772447', '3': '2023-06-13 12:36:03.772447', '4': '2023-06-13 12:36:04.772447'}}]
    result = list(stream.read_records(sync_mode=SyncMode.incremental, cursor_field=['change_status.last_change_date_time'], stream_slice=stream_slices[0]))
    expected_result = [{'campaign.id': 1, 'campaign_criterion.resource_name': '1', 'change_status.last_change_date_time': '2023-06-13 12:36:01.772447', 'customer.id': 123}, {'campaign.id': 1, 'campaign_criterion.resource_name': '2', 'change_status.last_change_date_time': '2023-06-13 12:36:02.772447', 'customer.id': 123}, {'campaign_criterion.resource_name': '3', 'deleted_at': '2023-06-13 12:36:03.772447'}, {'campaign_criterion.resource_name': '4', 'deleted_at': '2023-06-13 12:36:04.772447'}]
    assert all([expected_row in result for expected_row in expected_result])
    assert stream.state == {'change_status': {'123': {'change_status.last_change_date_time': '2023-06-13 12:36:04.772447'}}}
    assert stream.get_query.call_count == 1

def mock_response_1():
    if False:
        return 10
    yield [{'change_status.last_change_date_time': '2023-06-13 12:36:01.772447', 'change_status.resource_type': 'CAMPAIGN_CRITERION', 'change_status.resource_status': 'ADDED', 'change_status.campaign_criterion': '1'}, {'change_status.last_change_date_time': '2023-06-13 12:36:02.772447', 'change_status.resource_type': 'CAMPAIGN_CRITERION', 'change_status.resource_status': 'ADDED', 'change_status.campaign_criterion': '2'}]

def mock_response_2():
    if False:
        i = 10
        return i + 15
    yield [{'change_status.last_change_date_time': '2023-06-13 12:36:03.772447', 'change_status.resource_type': 'CAMPAIGN_CRITERION', 'change_status.resource_status': 'REMOVED', 'change_status.campaign_criterion': '3'}, {'change_status.last_change_date_time': '2023-06-13 12:36:04.772447', 'change_status.resource_type': 'CAMPAIGN_CRITERION', 'change_status.resource_status': 'REMOVED', 'change_status.campaign_criterion': '4'}]

def mock_response_3():
    if False:
        return 10
    yield [{'change_status.last_change_date_time': '2023-06-13 12:36:04.772447', 'change_status.resource_type': 'CAMPAIGN_CRITERION', 'change_status.resource_status': 'REMOVED', 'change_status.campaign_criterion': '6'}]

def mock_response_4():
    if False:
        i = 10
        return i + 15
    yield [{'change_status.last_change_date_time': '2023-06-13 12:36:04.772447', 'change_status.resource_type': 'CAMPAIGN_CRITERION', 'change_status.resource_status': 'REMOVED', 'change_status.campaign_criterion': '6'}, {'change_status.last_change_date_time': '2023-06-13 12:36:04.772447', 'change_status.resource_type': 'CAMPAIGN_CRITERION', 'change_status.resource_status': 'REMOVED', 'change_status.campaign_criterion': '7'}]

class MockGoogleAdsLimit(GoogleAds):
    count = 0

    def parse_single_result(self, schema, result):
        if False:
            print('Hello World!')
        return result

    def send_request(self, query: str, customer_id: str):
        if False:
            print('Hello World!')
        self.count += 1
        if self.count == 1:
            return mock_response_1()
        elif self.count == 2:
            return mock_response_2()
        else:
            return mock_response_3()

def mock_query_limit(self) -> int:
    if False:
        print('Hello World!')
    return 2

def copy_call_args(mock):
    if False:
        i = 10
        return i + 15
    new_mock = Mock()

    def side_effect(*args, **kwargs):
        if False:
            while True:
                i = 10
        args = deepcopy(args)
        kwargs = deepcopy(kwargs)
        new_mock(*args, **kwargs)
        return DEFAULT
    mock.side_effect = side_effect
    return new_mock

def test_query_limit_hit(config, customers):
    if False:
        print('Hello World!')
    '\n    Test the behavior of the `read_records` method in the `ChangeStatus` stream when the query limit is hit.\n\n    This test simulates a scenario where the limit is hit and slice start_date is updated with latest record cursor\n    '
    customer_id = next(iter(customers)).id
    stream_slice = {'customer_id': customer_id, 'start_date': '2023-06-13 11:35:04.772447', 'end_date': '2023-06-13 13:36:04.772447'}
    google_api = MockGoogleAdsLimit(credentials=config['credentials'])
    stream_config = dict(api=google_api, customers=customers)
    stream = ChangeStatus(**stream_config)
    ChangeStatus.query_limit = property(mock_query_limit)
    stream.get_query = Mock(return_value='query')
    get_query_mock = copy_call_args(stream.get_query)
    result = list(stream.read_records(sync_mode=SyncMode.incremental, cursor_field=['change_status.last_change_date_time'], stream_slice=stream_slice))
    assert len(result) == 5
    assert stream.get_query.call_count == 3
    get_query_calls = [call({'customer_id': '123', 'start_date': '2023-06-13 11:35:04.772447', 'end_date': '2023-06-13 13:36:04.772447'}), call({'customer_id': '123', 'start_date': '2023-06-13 12:36:02.772447', 'end_date': '2023-06-13 13:36:04.772447'}), call({'customer_id': '123', 'start_date': '2023-06-13 12:36:04.772447', 'end_date': '2023-06-13 13:36:04.772447'})]
    get_query_mock.assert_has_calls(get_query_calls)

class MockGoogleAdsLimitException(MockGoogleAdsLimit):

    def send_request(self, query: str, customer_id: str):
        if False:
            return 10
        self.count += 1
        if self.count == 1:
            return mock_response_1()
        elif self.count == 2:
            return mock_response_2()
        elif self.count == 3:
            return mock_response_4()

def test_query_limit_hit_exception(config, customers):
    if False:
        return 10
    '\n    Test the behavior of the `read_records` method in the `ChangeStatus` stream when the query limit is hit.\n\n    This test simulates a scenario where the limit is hit and there are more than query_limit number of records with same cursor,\n    then error will be raised\n    '
    customer_id = next(iter(customers)).id
    stream_slice = {'customer_id': customer_id, 'start_date': '2023-06-13 11:35:04.772447', 'end_date': '2023-06-13 13:36:04.772447'}
    google_api = MockGoogleAdsLimitException(credentials=config['credentials'])
    stream_config = dict(api=google_api, customers=customers)
    stream = ChangeStatus(**stream_config)
    ChangeStatus.query_limit = property(mock_query_limit)
    stream.get_query = Mock(return_value='query')
    with pytest.raises(AirbyteTracedException) as e:
        list(stream.read_records(sync_mode=SyncMode.incremental, cursor_field=['change_status.last_change_date_time'], stream_slice=stream_slice))
    expected_message = 'More then limit 2 records with same cursor field. Incremental sync is not possible for this stream.'
    assert e.value.message == expected_message

def test_change_status_get_query(mocker, config, customers):
    if False:
        print('Hello World!')
    '\n    Test the get_query method of ChangeStatus stream.\n\n    Given a sample stream_slice, it verifies that the returned query is as expected.\n    '
    google_api = MockGoogleAds(credentials=config['credentials'])
    stream = ChangeStatus(api=google_api, customers=customers)
    mocker.patch.object(stream, 'get_json_schema', return_value={'properties': {'change_status.resource_type': {'type': 'str'}}})
    stream_slice = {'start_date': '2023-01-01 00:00:00.000000', 'end_date': '2023-09-19 00:00:00.000000', 'resource_type': 'SOME_RESOURCE_TYPE'}
    query = stream.get_query(stream_slice=stream_slice)
    expected_query = "SELECT change_status.resource_type FROM change_status WHERE change_status.last_change_date_time >= '2023-01-01 00:00:00.000000' AND change_status.last_change_date_time <= '2023-09-19 00:00:00.000000' AND change_status.resource_type = 'SOME_RESOURCE_TYPE' ORDER BY change_status.last_change_date_time ASC LIMIT 2"
    assert query == expected_query

def are_queries_equivalent(query1, query2):
    if False:
        return 10
    criteria1 = query1.split('IN (')[1].rstrip(')').split(', ')
    criteria2 = query2.split('IN (')[1].rstrip(')').split(', ')
    criteria1_sorted = sorted(criteria1)
    criteria2_sorted = sorted(criteria2)
    query1_sorted = query1.replace(', '.join(criteria1), ', '.join(criteria1_sorted))
    query2_sorted = query2.replace(', '.join(criteria2), ', '.join(criteria2_sorted))
    return query1_sorted == query2_sorted

def test_incremental_events_stream_get_query(mocker, config, customers):
    if False:
        i = 10
        return i + 15
    '\n    Test the get_query method of the IncrementalEventsStream class.\n\n    Given a sample stream_slice, this test will verify that the returned query string is as expected.\n    '
    google_api = MockGoogleAds(credentials=config['credentials'])
    stream = CampaignCriterion(api=google_api, customers=customers)
    mocker.patch.object(stream, 'get_json_schema', return_value={'properties': {'campaign_criterion.resource_name': {'type': 'str'}}})
    stream_slice = {'customer_id': '1234567890', 'updated_ids': {'customers/1234567890/adGroupCriteria/111111111111~1', 'customers/1234567890/adGroupCriteria/111111111111~2', 'customers/1234567890/adGroupCriteria/111111111111~3'}, 'deleted_ids': {'customers/1234567890/adGroupCriteria/111111111111~4', 'customers/1234567890/adGroupCriteria/111111111111~5'}, 'record_changed_time_map': {'customers/1234567890/adGroupCriteria/111111111111~1': '2023-09-18 08:56:53.413023', 'customers/1234567890/adGroupCriteria/111111111111~2': '2023-09-18 08:56:59.165599', 'customers/1234567890/adGroupCriteria/111111111111~3': '2023-09-18 08:56:59.165599', 'customers/1234567890/adGroupCriteria/111111111111~4': '2023-09-18 08:56:59.165599', 'customers/1234567890/adGroupCriteria/111111111111~5': '2023-09-18 08:56:59.165599'}}
    query = stream.get_query(stream_slice=stream_slice)
    expected_query = "SELECT campaign_criterion.resource_name FROM campaign_criterion WHERE campaign_criterion.resource_name IN ('customers/1234567890/adGroupCriteria/111111111111~1', 'customers/1234567890/adGroupCriteria/111111111111~2', 'customers/1234567890/adGroupCriteria/111111111111~3')"
    assert are_queries_equivalent(query, expected_query)

def test_read_records_with_slice_splitting(mocker, config):
    if False:
        return 10
    "\n    Test the read_records method to ensure it correctly splits the stream_slice and calls the parent's read_records.\n    "
    stream_slice = {'updated_ids': set(range(15000)), 'record_changed_time_map': {i: f'time_{i}' for i in range(15000)}, 'customer_id': 'sample_customer_id', 'deleted_ids': set()}
    google_api = MockGoogleAds(credentials=config['credentials'])
    stream = CampaignCriterion(api=google_api, customers=[])
    super_read_records_mock = MagicMock()
    mocker.patch('source_google_ads.streams.GoogleAdsStream.read_records', super_read_records_mock)
    read_deleted_records_mock = mocker.patch.object(stream, '_read_deleted_records', return_value=[])
    update_state_mock = mocker.patch.object(stream, '_update_state')
    list(stream.read_records(SyncMode.incremental, stream_slice=stream_slice))
    assert super_read_records_mock.call_count == 2
    expected_first_slice = {'updated_ids': set(range(10000)), 'record_changed_time_map': {i: f'time_{i}' for i in range(10000)}, 'customer_id': 'sample_customer_id', 'deleted_ids': set()}
    expected_second_slice = {'updated_ids': set(range(10000, 15000)), 'record_changed_time_map': {i: f'time_{i}' for i in range(10000, 15000)}, 'customer_id': 'sample_customer_id', 'deleted_ids': set()}
    (first_call_args, first_call_kwargs) = super_read_records_mock.call_args_list[0]
    assert first_call_args[0] == SyncMode.incremental
    assert first_call_kwargs['stream_slice'] == expected_first_slice
    (second_call_args, second_call_kwargs) = super_read_records_mock.call_args_list[1]
    assert second_call_args[0] == SyncMode.incremental
    assert second_call_kwargs['stream_slice'] == expected_second_slice
    read_deleted_records_mock.assert_called_once_with(stream_slice)
    update_state_mock.assert_called_once_with(stream_slice)

def test_update_state_with_parent_state(mocker):
    if False:
        while True:
            i = 10
    '\n    Test the _update_state method when the parent_stream has a state.\n    '
    stream = CampaignCriterion(api=MagicMock(), customers=[])
    stream.parent_stream.state = {'customer_id_1': {'change_status.last_change_date_time': '2023-10-20 00:00:00.000000'}}
    stream_slice_first = {'customer_id': 'customer_id_1'}
    stream._update_state(stream_slice_first)
    expected_state_first_call = {'change_status': {'customer_id_1': {'change_status.last_change_date_time': '2023-10-20 00:00:00.000000'}}}
    assert stream._state == expected_state_first_call
    stream.parent_stream.state = {'customer_id_2': {'change_status.last_change_date_time': '2023-10-21 00:00:00.000000'}}
    stream_slice_second = {'customer_id': 'customer_id_2'}
    stream._update_state(stream_slice_second)
    expected_state_second_call = {'change_status': {'customer_id_1': {'change_status.last_change_date_time': '2023-10-20 00:00:00.000000'}, 'customer_id_2': {'change_status.last_change_date_time': '2023-10-21 00:00:00.000000'}}}
    assert stream._state == expected_state_second_call
    now = pendulum.datetime(2023, 11, 2, 12, 53, 7)
    pendulum.set_test_now(now)
    stream_slice_third = {'customer_id': 'customer_id_3'}
    stream._update_state(stream_slice_third)
    expected_state_third_call = {'change_status': {'customer_id_1': {'change_status.last_change_date_time': '2023-10-20 00:00:00.000000'}, 'customer_id_2': {'change_status.last_change_date_time': '2023-10-21 00:00:00.000000'}, 'customer_id_3': {'change_status.last_change_date_time': '2023-11-02 00:00:00.000000'}}}
    assert stream._state == expected_state_third_call
    pendulum.set_test_now()

def test_update_state_without_parent_state(mocker):
    if False:
        return 10
    '\n    Test the _update_state method when the parent_stream does not have a state.\n    '
    pendulum.set_test_now()
    stream = CampaignCriterion(api=MagicMock(), customers=[])
    now = pendulum.datetime(2023, 11, 2, 12, 53, 7)
    pendulum.set_test_now(now)
    stream_slice_first = {'customer_id': 'customer_id_1'}
    stream._update_state(stream_slice_first)
    expected_state_first_call = {'change_status': {'customer_id_1': {'change_status.last_change_date_time': '2023-11-02 00:00:00.000000'}}}
    assert stream._state == expected_state_first_call
    stream_slice_second = {'customer_id': 'customer_id_2'}
    stream._update_state(stream_slice_second)
    expected_state_second_call = {'change_status': {'customer_id_1': {'change_status.last_change_date_time': '2023-11-02 00:00:00.000000'}, 'customer_id_2': {'change_status.last_change_date_time': '2023-11-02 00:00:00.000000'}}}
    assert stream._state == expected_state_second_call
    pendulum.set_test_now()