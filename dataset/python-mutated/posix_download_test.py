import backoff
from google.api_core.exceptions import RetryError
from google.api_core.exceptions import ServiceUnavailable
from google.cloud.storage import Bucket
import posix_download

@backoff.on_exception(backoff.expo, (RetryError, ServiceUnavailable), max_time=60)
def test_posix_download(capsys, project_id: str, job_description_unique: str, agent_pool_name: str, posix_root_directory: str, source_bucket: Bucket):
    if False:
        return 10
    posix_download.download_from_gcs(project_id=project_id, description=job_description_unique, sink_agent_pool_name=agent_pool_name, root_directory=posix_root_directory, source_bucket=source_bucket.name, gcs_source_path=posix_root_directory)
    (out, _) = capsys.readouterr()
    assert 'Created transferJob' in out