"""
Command-line sample that list operations for a transfer job.
"""
import argparse
import json
from google.cloud import storage_transfer

def transfer_check(project_id: str, job_name: str):
    if False:
        return 10
    '\n    Lists operations for a transfer job.\n    '
    client = storage_transfer.StorageTransferServiceClient()
    job_filter = json.dumps({'project_id': project_id, 'job_names': [job_name]})
    response = client.transport.operations_client.list_operations('transferOperations', job_filter)
    operations = [storage_transfer.TransferOperation.deserialize(item.metadata.value) for item in response]
    print(f'Transfer operations for {job_name}`:', operations)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-id', help='The ID of the Google Cloud Platform Project that owns the job', required=True)
    parser.add_argument('--job-name', help='The transfer job to get', required=True)
    args = parser.parse_args()
    transfer_check(**vars(args))