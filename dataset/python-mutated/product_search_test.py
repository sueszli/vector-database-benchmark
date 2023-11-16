import os
import pytest
from product_search import get_similar_products_file, get_similar_products_uri
PROJECT_ID = os.getenv('GOOGLE_CLOUD_PROJECT')
LOCATION = 'us-west1'
PRODUCT_SET_ID = 'indexed_product_set_id_for_testing'
PRODUCT_CATEGORY = 'apparel'
PRODUCT_ID_1 = 'indexed_product_id_for_testing_1'
PRODUCT_ID_2 = 'indexed_product_id_for_testing_2'
FILE_PATH_1 = 'resources/shoes_1.jpg'
IMAGE_URI_1 = 'gs://cloud-samples-data/vision/product_search/shoes_1.jpg'
FILTER = 'style=womens'
MAX_RESULTS = 6

@pytest.mark.flaky(max_runs=5, min_passes=1)
def test_get_similar_products_file(capsys):
    if False:
        i = 10
        return i + 15
    get_similar_products_file(PROJECT_ID, LOCATION, PRODUCT_SET_ID, PRODUCT_CATEGORY, FILE_PATH_1, '', MAX_RESULTS)
    (out, _) = capsys.readouterr()
    assert PRODUCT_ID_1 in out
    assert PRODUCT_ID_2 in out

def test_get_similar_products_uri(capsys):
    if False:
        i = 10
        return i + 15
    get_similar_products_uri(PROJECT_ID, LOCATION, PRODUCT_SET_ID, PRODUCT_CATEGORY, IMAGE_URI_1, '')
    (out, _) = capsys.readouterr()
    assert PRODUCT_ID_1 in out
    assert PRODUCT_ID_2 in out

def test_get_similar_products_file_with_filter(capsys):
    if False:
        i = 10
        return i + 15
    get_similar_products_file(PROJECT_ID, LOCATION, PRODUCT_SET_ID, PRODUCT_CATEGORY, FILE_PATH_1, FILTER, MAX_RESULTS)
    (out, _) = capsys.readouterr()
    assert PRODUCT_ID_1 in out
    assert PRODUCT_ID_2 not in out

def test_get_similar_products_uri_with_filter(capsys):
    if False:
        for i in range(10):
            print('nop')
    get_similar_products_uri(PROJECT_ID, LOCATION, PRODUCT_SET_ID, PRODUCT_CATEGORY, IMAGE_URI_1, FILTER)
    (out, _) = capsys.readouterr()
    assert PRODUCT_ID_1 in out
    assert PRODUCT_ID_2 not in out