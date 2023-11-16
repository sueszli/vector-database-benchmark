"""
Command-line sample that creates a daily migration from a GCS bucket to a
Nearline GCS bucket for objects untouched for 30 days.
"""
import argparse
from datetime import datetime
from google.cloud import storage_transfer
from google.protobuf.duration_pb2 import Duration

def create_daily_nearline_30_day_migration(project_id: str, description: str, source_bucket: str, sink_bucket: str, start_date: datetime):
    if False:
        print('Hello World!')
    'Create a daily migration from a GCS bucket to a Nearline GCS bucket\n    for objects untouched for 30 days.'
    client = storage_transfer.StorageTransferServiceClient()
    transfer_job_request = storage_transfer.CreateTransferJobRequest({'transfer_job': {'project_id': project_id, 'description': description, 'status': storage_transfer.TransferJob.Status.ENABLED, 'schedule': {'schedule_start_date': {'day': start_date.day, 'month': start_date.month, 'year': start_date.year}}, 'transfer_spec': {'gcs_data_source': {'bucket_name': source_bucket}, 'gcs_data_sink': {'bucket_name': sink_bucket}, 'object_conditions': {'min_time_elapsed_since_last_modification': Duration(seconds=2592000)}, 'transfer_options': {'delete_objects_from_source_after_transfer': True}}}})
    result = client.create_transfer_job(transfer_job_request)
    print(f'Created transferJob: {result.name}')
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-id', help='The ID of the Google Cloud Platform Project that owns the job', required=True)
    parser.add_argument('--description', help='A useful description for your transfer job', required=True)
    parser.add_argument('--source-bucket', help='Google Cloud Storage source bucket name', required=True)
    parser.add_argument('--sink-bucket', help='Google Cloud Storage destination bucket name', required=True)
    args = parser.parse_args()
    create_daily_nearline_30_day_migration(start_date=datetime.utcnow(), **vars(args))