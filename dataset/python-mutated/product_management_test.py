import os
import uuid
import pytest
from product_management import create_product, delete_product, list_products, purge_orphan_products, update_product_labels
PROJECT_ID = os.getenv('GOOGLE_CLOUD_PROJECT')
LOCATION = 'us-west1'
PRODUCT_DISPLAY_NAME = 'fake_product_display_name_for_testing'
PRODUCT_CATEGORY = 'homegoods'
PRODUCT_ID = f'test_{uuid.uuid4()}'
KEY = 'fake_key_for_testing'
VALUE = 'fake_value_for_testing'

@pytest.fixture(scope='function', autouse=True)
def setup_teardown():
    if False:
        i = 10
        return i + 15
    create_product(PROJECT_ID, LOCATION, PRODUCT_ID, PRODUCT_DISPLAY_NAME, PRODUCT_CATEGORY)
    yield None
    delete_product(PROJECT_ID, LOCATION, PRODUCT_ID)

def test_delete_product(capsys):
    if False:
        while True:
            i = 10
    list_products(PROJECT_ID, LOCATION)
    (out, _) = capsys.readouterr()
    assert PRODUCT_ID in out
    delete_product(PROJECT_ID, LOCATION, PRODUCT_ID)
    list_products(PROJECT_ID, LOCATION)
    (out, _) = capsys.readouterr()
    assert PRODUCT_ID not in out

def test_update_product_labels(capsys):
    if False:
        while True:
            i = 10
    update_product_labels(PROJECT_ID, LOCATION, PRODUCT_ID, KEY, VALUE)
    (out, _) = capsys.readouterr()
    assert KEY in out
    assert VALUE in out

def test_purge_orphan_products(capsys):
    if False:
        for i in range(10):
            print('nop')
    list_products(PROJECT_ID, LOCATION)
    (out, _) = capsys.readouterr()
    assert PRODUCT_ID in out
    purge_orphan_products(PROJECT_ID, LOCATION, force=True)
    list_products(PROJECT_ID, LOCATION)
    (out, _) = capsys.readouterr()
    assert PRODUCT_ID not in out