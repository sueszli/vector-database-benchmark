import re
import subprocess
from google.api_core.retry import Retry

@Retry()
def test_create_product():
    if False:
        i = 10
        return i + 15
    output = str(subprocess.check_output('python import_user_events_inline.py', shell=True))
    assert re.match('.*import user events from inline source request.*?parent: "projects/.*?/locations/global/catalogs/default_catalog.*', output)
    assert re.match('.*import user events from inline source request.*?input_config.*?user_event_inline_source.*', output)
    assert re.match('.*the operation was started.*?projects/.*?/locations/global/catalogs/default_catalog/operations/import-user-events.*', output)
    assert re.match('.*import user events operation is done.*', output)
    assert re.match('.*number of successfully imported events.*?3.*', output)
    assert re.match('.*number of failures during the importing.*?0.*', output)