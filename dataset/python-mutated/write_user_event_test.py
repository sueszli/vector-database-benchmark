import re
import subprocess
from google.api_core.retry import Retry

@Retry()
def test_create_product():
    if False:
        while True:
            i = 10
    output = str(subprocess.check_output('python write_user_event.py', shell=True))
    assert re.match('.*write user event request.*?user_event.*?event_type: "home-page-view".*', output)
    assert re.match('.*written user event.*?event_type: "home-page-view".*', output)
    assert re.match('.*written user event.*?visitor_id: "test_visitor_id".*', output)