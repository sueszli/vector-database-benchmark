import re
import subprocess
from google.api_core.retry import Retry

@Retry()
def test_create_product():
    if False:
        for i in range(10):
            print('nop')
    output = str(subprocess.check_output('python purge_user_event.py', shell=True))
    assert re.match('.*the user event is written.*', output)
    assert re.match('.*purge user events request.*?parent: "projects/.*?/locations/global/catalogs/default_catalog.*', output)
    assert re.match('.*purge user events request.*?filter: "visitorId=.*?test_visitor_id.*?".*', output)
    assert re.match('.*purge user events request.*?parent: "projects/.*?/locations/global/catalogs/default_catalog.*', output)
    assert re.match('.*purge user events request.*?force: true.*', output)
    assert re.match('.*the purge operation was started.*?projects/.*?/locations/global/catalogs/default_catalog/operations/purge-user-events.*', output)