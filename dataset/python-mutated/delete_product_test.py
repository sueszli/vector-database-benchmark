import re
import subprocess
from google.api_core.retry import Retry

@Retry()
def test_delete_product():
    if False:
        i = 10
        return i + 15
    output = str(subprocess.check_output('python delete_product.py', shell=True))
    assert re.match('.*delete product request.*', output)
    assert re.match('.*name: "projects/.+/locations/global/catalogs/default_catalog/branches/0/products/.*', output)
    assert re.match('.*deleting product projects/.+/locations/global/catalogs/default_catalog/branches/0/products/.*', output)
    assert re.match('.*product was deleted.*', output)