from samples.test_data_for_analytics import test_input_record, test_output_slices
from source_linkedin_ads.analytics import make_analytics_slices
TEST_KEY_VALUE_MAP = {'camp_id': 'id'}
TEST_START_DATE = '2021-08-01'
TEST_END_DATE = '2021-09-30'
TEST_REQUEST_PRAMS = {}

def test_make_analytics_slices():
    if False:
        for i in range(10):
            print('nop')
    assert list(make_analytics_slices(test_input_record, TEST_KEY_VALUE_MAP, TEST_START_DATE, TEST_END_DATE)) == test_output_slices