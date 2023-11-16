import os
import uuid
import pytest
import data_processing_helper
OUTPUT_FILENAME = f'test-processed-data-{uuid.uuid4()}.csv'

@pytest.fixture(autouse=True)
def teardown():
    if False:
        for i in range(10):
            print('nop')
    yield
    try:
        os.remove(OUTPUT_FILENAME)
    except FileNotFoundError:
        print('No file to delete')

def test_data_processing_helper():
    if False:
        while True:
            i = 10
    assert OUTPUT_FILENAME not in os.listdir()
    data_processing_helper.preprocess_station_data('ghcnd-stations.txt', OUTPUT_FILENAME)
    assert OUTPUT_FILENAME in os.listdir()