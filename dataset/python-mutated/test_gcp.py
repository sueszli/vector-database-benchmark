from dagster import JobDefinition
from docs_snippets.deploying.gcp.gcp_job import gcs_job

def test_gcs_job():
    if False:
        print('Hello World!')
    assert isinstance(gcs_job, JobDefinition)