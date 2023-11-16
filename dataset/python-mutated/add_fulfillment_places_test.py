import re
import subprocess
from google.api_core.retry import Retry

@Retry()
def test_add_fulfillment():
    if False:
        i = 10
        return i + 15
    output = str(subprocess.check_output('python add_fulfillment_places.py', shell=True))
    assert re.match('.*product is created.*', output)
    assert re.match('.*add fulfillment request.*', output)
    assert re.match('.*add fulfillment places.*', output)
    assert re.match('.*get product response.*?fulfillment_info.*type_: "pickup-in-store".*?place_ids: "store1".*', output)
    assert re.match('.*get product response.*?fulfillment_info.*type_: "pickup-in-store".*?place_ids: "store2".*', output)