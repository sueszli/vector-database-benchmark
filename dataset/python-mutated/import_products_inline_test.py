import re
import subprocess
from google.api_core.retry import Retry

@Retry()
def test_import_products_gcs():
    if False:
        for i in range(10):
            print('nop')
    output = str(subprocess.check_output('python import_products_inline_source.py', shell=True))
    assert re.match('.*import products from inline source request.*', output)
    assert re.match('.*the operation was started.*', output)
    assert re.match('.*projects/.*/locations/global/catalogs/default_catalog/branches/0/operations/import-products.*', output)
    assert re.match('.*number of successfully imported products.*?2.*', output)
    assert re.match('.*number of failures during the importing.*?0.*', output)