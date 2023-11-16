import os
import uuid
import pytest
from product_management import create_product, delete_product
from reference_image_management import create_reference_image, delete_reference_image, list_reference_images
PROJECT_ID = os.getenv('GOOGLE_CLOUD_PROJECT')
LOCATION = 'us-west1'
PRODUCT_DISPLAY_NAME = 'fake_product_display_name_for_testing'
PRODUCT_CATEGORY = 'homegoods'
PRODUCT_ID = f'test_{uuid.uuid4()}'
REFERENCE_IMAGE_ID = 'fake_reference_image_id_for_testing'
GCS_URI = 'gs://cloud-samples-data/vision/product_search/shoes_1.jpg'

@pytest.fixture(scope='function', autouse=True)
def setup_teardown():
    if False:
        i = 10
        return i + 15
    create_product(PROJECT_ID, LOCATION, PRODUCT_ID, PRODUCT_DISPLAY_NAME, PRODUCT_CATEGORY)
    yield None
    delete_product(PROJECT_ID, LOCATION, PRODUCT_ID)

def test_create_reference_image(capsys):
    if False:
        return 10
    create_reference_image(PROJECT_ID, LOCATION, PRODUCT_ID, REFERENCE_IMAGE_ID, GCS_URI)
    list_reference_images(PROJECT_ID, LOCATION, PRODUCT_ID)
    (out, _) = capsys.readouterr()
    assert REFERENCE_IMAGE_ID in out

def test_delete_reference_image(capsys):
    if False:
        for i in range(10):
            print('nop')
    create_reference_image(PROJECT_ID, LOCATION, PRODUCT_ID, REFERENCE_IMAGE_ID, GCS_URI)
    list_reference_images(PROJECT_ID, LOCATION, PRODUCT_ID)
    (out, _) = capsys.readouterr()
    assert REFERENCE_IMAGE_ID in out
    delete_reference_image(PROJECT_ID, LOCATION, PRODUCT_ID, REFERENCE_IMAGE_ID)
    list_reference_images(PROJECT_ID, LOCATION, PRODUCT_ID)
    (out, _) = capsys.readouterr()
    assert REFERENCE_IMAGE_ID not in out