from datetime import date
import pendulum
import pytest
from airbyte_cdk.utils import AirbyteTracedException
from google.auth import exceptions
from source_google_ads.google_ads import GoogleAds
from source_google_ads.streams import chunk_date_range
from .common import MockGoogleAdsClient, MockGoogleAdsService
SAMPLE_SCHEMA = {'properties': {'segment.date': {'type': ['null', 'string']}}}

class MockedDateSegment:

    def __init__(self, date: str):
        if False:
            i = 10
            return i + 15
        self._mock_date = date

    def __getattr__(self, attr):
        if False:
            for i in range(10):
                print('nop')
        if attr == 'date':
            return date.fromisoformat(self._mock_date)
        return MockedDateSegment(self._mock_date)
SAMPLE_CONFIG = {'credentials': {'developer_token': 'developer_token', 'client_id': 'client_id', 'client_secret': 'client_secret', 'refresh_token': 'refresh_token'}}
EXPECTED_CRED = {'developer_token': 'developer_token', 'client_id': 'client_id', 'client_secret': 'client_secret', 'refresh_token': 'refresh_token', 'use_proto_plus': True}

def test_google_ads_init(mocker):
    if False:
        while True:
            i = 10
    google_client_mocker = mocker.patch('source_google_ads.google_ads.GoogleAdsClient', return_value=MockGoogleAdsClient)
    _ = GoogleAds(**SAMPLE_CONFIG)
    assert google_client_mocker.load_from_dict.call_args[0][0] == EXPECTED_CRED

def test_google_ads_wrong_permissions(mocker):
    if False:
        while True:
            i = 10
    mocker.patch('source_google_ads.google_ads.GoogleAdsClient.load_from_dict', side_effect=exceptions.RefreshError('invalid_grant'))
    with pytest.raises(AirbyteTracedException) as e:
        GoogleAds(**SAMPLE_CONFIG)
    expected_message = 'The authentication to Google Ads has expired. Re-authenticate to restore access to Google Ads.'
    assert e.value.message == expected_message

def test_send_request(mocker, customers):
    if False:
        print('Hello World!')
    mocker.patch('source_google_ads.google_ads.GoogleAdsClient.load_from_dict', return_value=MockGoogleAdsClient(SAMPLE_CONFIG))
    mocker.patch('source_google_ads.google_ads.GoogleAdsClient.get_service', return_value=MockGoogleAdsService())
    google_ads_client = GoogleAds(**SAMPLE_CONFIG)
    query = 'Query'
    page_size = 1000
    customer_id = next(iter(customers)).id
    response = list(google_ads_client.send_request(query, customer_id=customer_id))
    assert response[0].customer_id == customer_id
    assert response[0].query == query
    assert response[0].page_size == page_size

def test_get_fields_from_schema():
    if False:
        print('Hello World!')
    response = GoogleAds.get_fields_from_schema(SAMPLE_SCHEMA)
    assert response == ['segment.date']

def test_interval_chunking():
    if False:
        i = 10
        return i + 15
    mock_intervals = [{'start_date': '2021-06-17', 'end_date': '2021-06-26'}, {'start_date': '2021-06-27', 'end_date': '2021-07-06'}, {'start_date': '2021-07-07', 'end_date': '2021-07-16'}, {'start_date': '2021-07-17', 'end_date': '2021-07-26'}, {'start_date': '2021-07-27', 'end_date': '2021-08-05'}, {'start_date': '2021-08-06', 'end_date': '2021-08-10'}]
    intervals = list(chunk_date_range(start_date='2021-07-01', end_date='2021-08-10', conversion_window=14, slice_duration=pendulum.Duration(days=9), time_zone='UTC'))
    assert mock_intervals == intervals
generic_schema = {'properties': {'ad_group_id': {}, 'segments.date': {}, 'campaign_id': {}, 'account_id': {}}}

@pytest.mark.parametrize('fields, table_name, conditions, order_field, limit, expected_sql', ((['ad_group_id', 'segments.date', 'campaign_id', 'account_id'], 'ad_group_ad', ["segments.date >= '2020-01-01'", "segments.date <= '2020-01-10'"], 'segments.date', None, "SELECT ad_group_id, segments.date, campaign_id, account_id FROM ad_group_ad WHERE segments.date >= '2020-01-01' AND segments.date <= '2020-01-10' ORDER BY segments.date ASC"), (['ad_group_id', 'segments.date', 'campaign_id', 'account_id'], 'ad_group_ad', None, None, None, 'SELECT ad_group_id, segments.date, campaign_id, account_id FROM ad_group_ad'), (['ad_group_id', 'segments.date', 'campaign_id', 'account_id'], 'click_view', None, 'ad_group_id', 5, 'SELECT ad_group_id, segments.date, campaign_id, account_id FROM click_view ORDER BY ad_group_id ASC LIMIT 5')))
def test_convert_schema_into_query(fields, table_name, conditions, order_field, limit, expected_sql):
    if False:
        for i in range(10):
            print('nop')
    query = GoogleAds.convert_schema_into_query(fields, table_name, conditions, order_field, limit)
    assert query == expected_sql

def test_get_field_value():
    if False:
        i = 10
        return i + 15
    field = 'segment.date'
    date = '2001-01-01'
    response = GoogleAds.get_field_value(MockedDateSegment(date), field, {})
    assert response == date

def test_parse_single_result():
    if False:
        print('Hello World!')
    date = '2001-01-01'
    response = GoogleAds.parse_single_result(SAMPLE_SCHEMA, MockedDateSegment(date))
    assert response == response

def test_get_fields_metadata(mocker):
    if False:
        return 10
    mocker.patch('source_google_ads.google_ads.GoogleAdsClient', MockGoogleAdsClient)
    google_ads_client = GoogleAds(**SAMPLE_CONFIG)
    fields = ['field1', 'field2', 'field3']
    response = google_ads_client.get_fields_metadata(fields)
    mock_service = google_ads_client.client.get_service('GoogleAdsFieldService')
    expected_query = "\n        SELECT\n          name,\n          data_type,\n          enum_values,\n          is_repeated\n        WHERE name in ('field1','field2','field3')\n        "
    assert mock_service.request_query.strip() == expected_query.strip()
    assert set(response.keys()) == set(fields)
    for field in fields:
        assert response[field].name == field