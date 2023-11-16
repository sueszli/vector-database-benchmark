import re
import subprocess
from google.api_core.retry import Retry

@Retry()
def test_create_product():
    if False:
        print('Hello World!')
    output = str(subprocess.check_output('python rejoin_user_event.py', shell=True))
    assert re.match('.*the user event is written.*', output)
    assert re.match('.*rejoin user events request.*?parent: "projects/.*?/locations/global/catalogs/default_catalog.*', output)
    assert re.match('.*rejoin user events request.*?user_event_rejoin_scope: UNJOINED_EVENTS.*', output)
    assert re.match('.*the rejoin operation was started.*?projects/.*?/locations/global/catalogs/default_catalog/operations/rejoin-user-events.*', output)
    assert re.match('.*the purge operation was started.*?projects/.*?/locations/global/catalogs/default_catalog/operations/purge-user-events.*', output)