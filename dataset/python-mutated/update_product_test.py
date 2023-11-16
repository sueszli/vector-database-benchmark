import re
import subprocess
from google.api_core.retry import Retry

@Retry()
def test_add_fulfillment():
    if False:
        print('Hello World!')
    output = str(subprocess.check_output('python update_product.py', shell=True))
    assert re.match('.*product is created.*', output)
    assert re.match('.*updated product.*', output)
    assert re.match('.*updated product.*?title.*?Updated Nest Mini.*', output)
    assert re.match('.*updated product.*?brands.*?Updated Google.*', output)
    assert re.match('.*updated product.*?price.*?20.*', output)
    assert re.match('.*product.*was deleted.*', output)