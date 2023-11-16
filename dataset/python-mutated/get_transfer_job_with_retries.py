"""
Command-line sample that gets the latest transfer operation for a given
transfer job with request retry configuration.
"""
import argparse
from google.api_core.retry import Retry
from google.cloud import storage_transfer

def get_transfer_job_with_retries(project_id: str, job_name: str, max_retry_duration: float):
    if False:
        for i in range(10):
            print('nop')
    '\n    Check the latest transfer operation associated with a transfer job with\n    retries.\n    '
    client = storage_transfer.StorageTransferServiceClient()
    transfer_job = client.get_transfer_job({'project_id': project_id, 'job_name': job_name}, retry=Retry(maximum=max_retry_duration))
    print(f'Fetched transfer job: {transfer_job.name} with a max retry duration of {max_retry_duration}s')
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-id', help='The ID of the Google Cloud Platform Project that owns the job', required=True)
    parser.add_argument('--job-name', help='The transfer job to get', required=True)
    parser.add_argument('--max-retry-duration', help='The maximum amount of time to delay in seconds', type=float, default=60)
    args = parser.parse_args()
    get_transfer_job_with_retries(**vars(args))