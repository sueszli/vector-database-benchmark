import os
from google.api_core.retry import Retry
import enable_cmek

@Retry()
def test_enable_cmek() -> None:
    if False:
        i = 10
        return i + 15
    project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
    response = enable_cmek.enable_cmek(project_id, '')
    assert response.kms_key_name == ''