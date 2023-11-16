import os
import uuid
from google.cloud import storage
import pytest
from import_product_sets import import_product_sets
from product_in_product_set_management import list_products_in_product_set
from product_management import delete_product, list_products
from product_set_management import delete_product_set, list_product_sets
from reference_image_management import list_reference_images
PROJECT_ID = os.getenv('GOOGLE_CLOUD_PROJECT')
LOCATION = 'us-west1'
FILENAME = uuid.uuid4()
GCS_URI = f'gs://{PROJECT_ID}/vision/{FILENAME}.csv'
PRODUCT_SET_DISPLAY_NAME = 'fake_product_set_display_name_for_testing'
PRODUCT_SET_ID = f'test_{uuid.uuid4()}'
PRODUCT_ID_1 = f'test_{uuid.uuid4()}'
IMAGE_URI_1 = 'shoes_1.jpg'

@pytest.fixture(scope='function', autouse=True)
def setup_teardown():
    if False:
        print('Hello World!')
    client = storage.Client(project=PROJECT_ID)
    bucket = client.get_bucket(PROJECT_ID)
    blob = storage.Blob(f'vision/{FILENAME}.csv', bucket)
    blob.upload_from_string('"gs://cloud-samples-data/vision/product_search/shoes_1.jpg",' + f'"{IMAGE_URI_1}",' + f'"{PRODUCT_SET_ID}",' + f'"{PRODUCT_ID_1}",' + '"apparel",,"style=womens","0.1,0.1,0.9,0.1,0.9,0.9,0.1,0.9"')
    yield
    delete_product(PROJECT_ID, LOCATION, PRODUCT_ID_1)
    delete_product_set(PROJECT_ID, LOCATION, PRODUCT_SET_ID)
    blob.delete(client)

def test_import_product_sets(capsys):
    if False:
        i = 10
        return i + 15
    import_product_sets(PROJECT_ID, LOCATION, GCS_URI)
    list_product_sets(PROJECT_ID, LOCATION)
    (out, _) = capsys.readouterr()
    assert PRODUCT_SET_ID in out
    list_products(PROJECT_ID, LOCATION)
    (out, _) = capsys.readouterr()
    assert PRODUCT_ID_1 in out
    list_products_in_product_set(PROJECT_ID, LOCATION, PRODUCT_SET_ID)
    (out, _) = capsys.readouterr()
    assert PRODUCT_ID_1 in out
    list_reference_images(PROJECT_ID, LOCATION, PRODUCT_ID_1)
    (out, _) = capsys.readouterr()
    assert IMAGE_URI_1 in out