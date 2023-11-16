import os
import uuid
import pytest
from product_set_management import create_product_set, delete_product_set, list_product_sets
PROJECT_ID = os.getenv('GOOGLE_CLOUD_PROJECT')
LOCATION = 'us-west1'
PRODUCT_SET_DISPLAY_NAME = 'fake_product_set_display_name_for_testing'
PRODUCT_SET_ID = f'test_{uuid.uuid4()}'

@pytest.fixture(scope='function', autouse=True)
def teardown():
    if False:
        return 10
    yield
    delete_product_set(PROJECT_ID, LOCATION, PRODUCT_SET_ID)

def test_create_product_set(capsys):
    if False:
        while True:
            i = 10
    create_product_set(PROJECT_ID, LOCATION, PRODUCT_SET_ID, PRODUCT_SET_DISPLAY_NAME)
    list_product_sets(PROJECT_ID, LOCATION)
    (out, _) = capsys.readouterr()
    assert PRODUCT_SET_ID in out