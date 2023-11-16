import backoff
from google.api_core.exceptions import RetryError
from google.cloud.storage import Bucket
import manifest_request

@backoff.on_exception(backoff.expo, (RetryError,), max_time=60)
def test_manifest_request(capsys, project_id: str, job_description_unique: str, agent_pool_name: str, posix_root_directory: str, destination_bucket: Bucket, manifest_file: str):
    if False:
        for i in range(10):
            print('nop')
    manifest_request.create_transfer_with_manifest(project_id=project_id, description=job_description_unique, source_agent_pool_name=agent_pool_name, root_directory=posix_root_directory, sink_bucket=destination_bucket.name, manifest_location=manifest_file)
    (out, _) = capsys.readouterr()
    assert 'Created transferJob' in out