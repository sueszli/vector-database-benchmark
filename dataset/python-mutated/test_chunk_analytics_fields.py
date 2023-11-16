from source_linkedin_ads.analytics import chunk_analytics_fields
TEST_FIELDS_CHUNK_SIZE = 3
TEST_ANALYTICS_FIELDS = ['field_1', 'base_field_1', 'field_2', 'base_field_2', 'field_3', 'field_4', 'field_5', 'field_6', 'field_7', 'field_8']
TEST_BASE_ANALLYTICS_FIELDS = ['base_field_1', 'base_field_2']

def test_chunk_analytics_fields():
    if False:
        while True:
            i = 10
    '\n    We expect to truncate the fields list into the chunks of equal size,\n    with TEST_BASE_ANALLYTICS_FIELDS presence in each chunk,\n    order is not matter.\n    '
    expected_output = [['field_1', 'base_field_1', 'field_2', 'base_field_2'], ['base_field_2', 'field_3', 'field_4', 'base_field_1'], ['field_5', 'field_6', 'field_7', 'base_field_1', 'base_field_2'], ['field_8', 'base_field_1', 'base_field_2']]
    assert list(chunk_analytics_fields(TEST_ANALYTICS_FIELDS, TEST_BASE_ANALLYTICS_FIELDS, TEST_FIELDS_CHUNK_SIZE)) == expected_output