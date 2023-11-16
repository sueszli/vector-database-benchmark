import pytest
import requests
from source_google_analytics_data_api.api_quota import GoogleAnalyticsApiQuota
TEST_QUOTA_INSTANCE: GoogleAnalyticsApiQuota = GoogleAnalyticsApiQuota()

@pytest.fixture(name='expected_quota_list')
def expected_quota_list():
    if False:
        print('Hello World!')
    'The Quota were currently handle'
    return ['concurrentRequests', 'tokensPerProjectPerHour', 'potentiallyThresholdedRequestsPerHour']

def test_check_initial_quota_is_empty():
    if False:
        while True:
            i = 10
    '\n    Check the initial quota property is empty (== None), but ready to be fullfield.\n    '
    assert not TEST_QUOTA_INSTANCE.initial_quota

@pytest.mark.parametrize(('response_quota', 'partial_quota', 'should_retry_exp', 'backoff_time_exp', 'raise_on_http_errors_exp', 'stop_iter_exp'), [({'propertyQuota': {'concurrentRequests': {'consumed': 0, 'remaining': 10}, 'tokensPerProjectPerHour': {'consumed': 1, 'remaining': 1735}, 'potentiallyThresholdedRequestsPerHour': {'consumed': 1, 'remaining': 26}}}, False, True, None, True, False), ({'propertyQuota': {'concurrentRequests': {'consumed': 0, 'remaining': 10}, 'tokensPerProjectPerHour': {'consumed': 5, 'remaining': 955}, 'potentiallyThresholdedRequestsPerHour': {'consumed': 3, 'remaining': 26}}}, True, True, None, True, False), ({'propertyQuota': {'concurrentRequests': {'consumed': 2, 'remaining': 8}, 'tokensPerProjectPerHour': {'consumed': 5, 'remaining': 172}, 'potentiallyThresholdedRequestsPerHour': {'consumed': 3, 'remaining': 26}}}, True, True, 1800, False, False), ({'propertyQuota': {'concurrentRequests': {'consumed': 9, 'remaining': 1}, 'tokensPerProjectPerHour': {'consumed': 5, 'remaining': 935}, 'potentiallyThresholdedRequestsPerHour': {'consumed': 1, 'remaining': 26}}}, True, True, 30, False, False), ({'propertyQuota': {'concurrentRequests': {'consumed': 1, 'remaining': 9}, 'tokensPerProjectPerHour': {'consumed': 5, 'remaining': 935}, 'potentiallyThresholdedRequestsPerHour': {'consumed': 26, 'remaining': 2}}}, True, True, 1800, False, False)], ids=['Full', 'Partial', 'Running out tokensPerProjectPerHour', 'Running out concurrentRequests', 'Running out potentiallyThresholdedRequestsPerHour'])
def test_check_full_quota(requests_mock, expected_quota_list, response_quota, partial_quota, should_retry_exp, backoff_time_exp, raise_on_http_errors_exp, stop_iter_exp):
    if False:
        for i in range(10):
            print('nop')
    '\n    Check the quota and prepare the initial values for subsequent comparison with subsequent response calls.\n    The default values for the scenario are expected when the quota is full.\n    '
    url = 'https://analyticsdata.googleapis.com/v1beta/'
    payload = response_quota
    requests_mock.post(url, json=payload)
    response = requests.post(url)
    TEST_QUOTA_INSTANCE._check_quota(response)
    assert [quota in expected_quota_list for quota in TEST_QUOTA_INSTANCE.initial_quota.keys()]
    if partial_quota:
        current_quota = TEST_QUOTA_INSTANCE._get_known_quota_from_response(response.json().get('propertyQuota'))
        assert not current_quota == TEST_QUOTA_INSTANCE.initial_quota
    assert TEST_QUOTA_INSTANCE.should_retry is should_retry_exp
    assert TEST_QUOTA_INSTANCE.backoff_time == backoff_time_exp
    assert TEST_QUOTA_INSTANCE.raise_on_http_errors is raise_on_http_errors_exp
    assert TEST_QUOTA_INSTANCE.stop_iter is stop_iter_exp