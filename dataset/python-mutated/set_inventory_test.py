import re
import subprocess
from google.api_core.retry import Retry

@Retry()
def test_set_inventory():
    if False:
        print('Hello World!')
    output = str(subprocess.check_output('python set_inventory.py', shell=True))
    print(output)
    assert re.match('.*product is created.*', output)
    assert re.match('.*name: "projects/.*/locations/global/catalogs/default_catalog/branches/0/products.*', output)
    assert re.match('.*set inventory request.*', output)
    assert re.match('.*get product response.*?fulfillment_info.*type_: "pickup-in-store".*?place_ids: "store1".*', output)
    assert re.match('.*get product response.*?fulfillment_info.*type_: "pickup-in-store".*?place_ids: "store2".*', output)
    assert re.match('.*product projects/.*/locations/global/catalogs/default_catalog/branches/0/products.* was deleted.*', output)