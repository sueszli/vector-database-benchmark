import re
from _pytest.capture import CaptureFixture
import workload_identity_federation

def test_workload_identity_federation_aws(capsys: CaptureFixture) -> None:
    if False:
        return 10
    import google.auth
    (credentials, project_id) = google.auth.default()
    workload_identity_federation.create_token_aws(project_id, 'provider_id', 'pool_id')
    (out, _) = capsys.readouterr()
    assert re.search('URL encoded token:', out)