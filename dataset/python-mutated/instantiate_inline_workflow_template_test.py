import os
import backoff
from google.api_core.exceptions import Aborted, InternalServerError, ServiceUnavailable
import instantiate_inline_workflow_template
PROJECT_ID = os.environ['GOOGLE_CLOUD_PROJECT']
REGION = 'us-central1'

@backoff.on_exception(backoff.expo, (InternalServerError, ServiceUnavailable, Aborted), max_tries=5)
def test_workflows(capsys):
    if False:
        for i in range(10):
            print('nop')
    instantiate_inline_workflow_template.instantiate_inline_workflow_template(PROJECT_ID, REGION)
    (out, _) = capsys.readouterr()
    assert 'successfully' in out