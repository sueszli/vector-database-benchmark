import re
import subprocess
from google.api_core.retry import Retry

@Retry()
def test_add_fulfillment():
    if False:
        while True:
            i = 10
    output = str(subprocess.check_output('python remove_fulfillment_places.py', shell=True))
    assert re.match('.*product is created.*', output)
    assert re.match('.*remove fulfillment request.*', output)
    assert re.match('.*remove fulfillment places.*', output)
    assert re.match('.*get product response.*?fulfillment_info.*type_: "pickup-in-store".*?place_ids: "store1".*', output)
    assert not re.search('.*get product response.*?fulfillment_info.*store0.*', output)