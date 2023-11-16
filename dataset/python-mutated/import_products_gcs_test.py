import re
import subprocess
from google.api_core.retry import Retry
from setup_product.setup_cleanup import create_bucket, delete_bucket, upload_blob

@Retry()
def test_import_products_gcs(bucket_name_prefix):
    if False:
        while True:
            i = 10
    bucket_name = bucket_name_prefix[63:]
    try:
        create_bucket(bucket_name)
        upload_blob(bucket_name, '../resources/products.json')
        output = str(subprocess.check_output(f'python import_products_gcs.py {bucket_name}', shell=True))
    finally:
        delete_bucket(bucket_name)
    assert re.match('.*import products from google cloud source request.*', output)
    assert re.match('.*input_uris: "gs://.*/products.json".*', output)
    assert re.match('.*the operation was started.*', output)
    assert re.match('.*projects/.*/locations/global/catalogs/default_catalog/branches/0/operations/import-products.*', output)
    assert re.match('.*number of successfully imported products.*?316.*', output)
    assert re.match('.*number of failures during the importing.*?0.*', output)