import os
import uuid
import pytest
from product_set_management import create_product_set, delete_product_set, list_product_sets
PROJECT_ID = os.getenv('GOOGLE_CLOUD_PROJECT')
LOCATION = 'us-west1'
PRODUCT_SET_DISPLAY_NAME = 'fake_product_set_display_name_for_testing'
PRODUCT_SET_ID = f'test_{uuid.uuid4()}'

@pytest.fixture(scope='function', autouse=True)
def setup():
    if False:
        for i in range(10):
            print('nop')
    create_product_set(PROJECT_ID, LOCATION, PRODUCT_SET_ID, PRODUCT_SET_DISPLAY_NAME)

def test_delete_product_set(capsys):
    if False:
        print('Hello World!')
    list_product_sets(PROJECT_ID, LOCATION)
    (out, _) = capsys.readouterr()
    assert PRODUCT_SET_ID in out
    delete_product_set(PROJECT_ID, LOCATION, PRODUCT_SET_ID)
    list_product_sets(PROJECT_ID, LOCATION)
    (out, _) = capsys.readouterr()
    assert PRODUCT_SET_ID not in out