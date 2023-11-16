"""
Command-line sample that creates aan event driven transfer between two GCS buckets that tracks a PubSub subscription.
"""
import argparse
from google.cloud import storage_transfer

def create_event_driven_gcs_transfer(project_id: str, description: str, source_bucket: str, sink_bucket: str, pubsub_id: str):
    if False:
        i = 10
        return i + 15
    'Create an event driven transfer between two GCS buckets that tracks a PubSub subscription'
    client = storage_transfer.StorageTransferServiceClient()
    transfer_job_request = storage_transfer.CreateTransferJobRequest({'transfer_job': {'project_id': project_id, 'description': description, 'status': storage_transfer.TransferJob.Status.ENABLED, 'transfer_spec': {'gcs_data_source': {'bucket_name': source_bucket}, 'gcs_data_sink': {'bucket_name': sink_bucket}}, 'event_stream': {'name': pubsub_id}}})
    result = client.create_transfer_job(transfer_job_request)
    print(f'Created transferJob: {result.name}')
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-id', help='The ID of the Google Cloud Platform Project that owns the job', required=True)
    parser.add_argument('--description', help='A useful description for your transfer job', default='My transfer job')
    parser.add_argument('--source-bucket', help='Google Cloud Storage source bucket name', required=True)
    parser.add_argument('--sink-bucket', help='Google Cloud Storage destination bucket name', required=True)
    parser.add_argument('--pubsub-id', help='The subscription ID of the PubSub queue to track', required=True)
    args = parser.parse_args()
    create_event_driven_gcs_transfer(**vars(args))