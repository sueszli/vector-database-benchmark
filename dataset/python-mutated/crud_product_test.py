import re
import subprocess
from google.api_core.retry import Retry

@Retry()
def test_add_fulfillment():
    if False:
        print('Hello World!')
    output = str(subprocess.check_output('python crud_product.py', shell=True))
    assert re.match('.*create product request.*', output)
    assert re.match('.*create product request.*', output)
    assert re.match('.*create product request.*?title: "Nest Mini".*', output)
    assert re.match('.*created product.*', output)
    assert re.match('.*created product.*?title: "Nest Mini".*', output)
    assert re.match('.*get product request.*', output)
    assert re.match('.*get product response.*', output)
    assert re.match('.*update product request.*', output)
    assert re.match('.*update product request.*?title: "Updated Nest Mini.*', output)
    assert re.match('.*updated product.*?title.*?Updated Nest Mini.*', output)
    assert re.match('.*updated product.*?brands.*?Updated Google.*', output)
    assert re.match('.*updated product.*?price.*?20.*', output)
    assert re.match('.*product was deleted.*', output)