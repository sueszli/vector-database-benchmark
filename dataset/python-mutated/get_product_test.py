import re
import subprocess
from google.api_core.retry import Retry

@Retry()
def test_get_product():
    if False:
        return 10
    output = str(subprocess.check_output('python get_product.py', shell=True))
    assert re.match('.*get product request.*', output)
    assert re.match('.*get product response.*', output)
    assert re.match('.*get product response.*?name.*?projects/.*/locations/global/catalogs/default_catalog/branches/0/products/.*', output)
    assert re.match('.*get product response.*?title.*?Nest Mini.*', output)
    assert re.match('.*product.*was deleted.*', output)