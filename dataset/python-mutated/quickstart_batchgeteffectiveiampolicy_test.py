import os
from google.api_core import retry
from google.api_core.exceptions import DeadlineExceeded
import quickstart_batchgeteffectiveiampolicy
PROJECT = os.environ['GOOGLE_CLOUD_PROJECT']

@retry.Retry(retry.if_exception_type(DeadlineExceeded))
def test_batch_get_effective_iam_policies(capsys):
    if False:
        print('Hello World!')
    scope = f'projects/{PROJECT}'
    resource_names = [f'//cloudresourcemanager.googleapis.com/projects/{PROJECT}']
    quickstart_batchgeteffectiveiampolicy.batch_get_effective_iam_policies(resource_names, scope)
    (out, _) = capsys.readouterr()
    assert resource_names[0] in out