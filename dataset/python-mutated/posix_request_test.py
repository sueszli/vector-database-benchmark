import backoff
from google.api_core.exceptions import RetryError
from google.api_core.exceptions import ServiceUnavailable
from google.cloud.storage import Bucket
import posix_request

@backoff.on_exception(backoff.expo, (RetryError, ServiceUnavailable), max_time=60)
def test_posix_request(capsys, project_id: str, job_description_unique: str, agent_pool_name: str, posix_root_directory: str, destination_bucket: Bucket):
    if False:
        for i in range(10):
            print('nop')
    posix_request.transfer_from_posix_to_gcs(project_id=project_id, description=job_description_unique, source_agent_pool_name=agent_pool_name, root_directory=posix_root_directory, sink_bucket=destination_bucket.name)
    (out, _) = capsys.readouterr()
    assert 'Created transferJob' in out