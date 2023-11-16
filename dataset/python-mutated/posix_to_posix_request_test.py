import backoff
from google.api_core.exceptions import RetryError
from google.api_core.exceptions import ServiceUnavailable
from google.cloud.storage import Bucket
import posix_to_posix_request

@backoff.on_exception(backoff.expo, (RetryError, ServiceUnavailable), max_time=60)
def test_posix_to_posix_request(capsys, project_id: str, job_description_unique: str, agent_pool_name: str, posix_root_directory: str, intermediate_bucket: Bucket):
    if False:
        while True:
            i = 10
    posix_to_posix_request.transfer_between_posix(project_id=project_id, description=job_description_unique, source_agent_pool_name=agent_pool_name, sink_agent_pool_name=agent_pool_name, root_directory=posix_root_directory, destination_directory=posix_root_directory, intermediate_bucket=intermediate_bucket.name)
    (out, _) = capsys.readouterr()
    assert 'Created transferJob' in out