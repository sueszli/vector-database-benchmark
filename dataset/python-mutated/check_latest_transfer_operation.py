"""
Command-line sample that checks the latest transfer operation for a given
transfer job.
"""
import argparse
from google.cloud import storage_transfer

def check_latest_transfer_operation(project_id: str, job_name: str):
    if False:
        print('Hello World!')
    'Checks the latest transfer operation for a given transfer job.'
    client = storage_transfer.StorageTransferServiceClient()
    transfer_job = client.get_transfer_job({'project_id': project_id, 'job_name': job_name})
    if transfer_job.latest_operation_name:
        response = client.transport.operations_client.get_operation(transfer_job.latest_operation_name)
        operation = storage_transfer.TransferOperation.deserialize(response.metadata.value)
        print(f'Latest transfer operation for `{job_name}`: {operation}')
    else:
        print(f'Transfer job {job_name} has not ran yet.')
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-id', help='The ID of the Google Cloud Platform Project that owns the job', required=True)
    parser.add_argument('--job-name', help='The transfer job to get', required=True)
    args = parser.parse_args()
    check_latest_transfer_operation(**vars(args))