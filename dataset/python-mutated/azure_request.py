"""
Command-line sample that creates a one-time transfer from Azure Blob Storage to
Google Cloud Storage.
"""
from datetime import datetime
from google.cloud import storage_transfer

def create_one_time_azure_transfer(project_id: str, description: str, azure_storage_account: str, azure_sas_token: str, source_container: str, sink_bucket: str):
    if False:
        for i in range(10):
            print('nop')
    'Creates a one-time transfer job from Azure Blob Storage to Google Cloud\n    Storage.'
    client = storage_transfer.StorageTransferServiceClient()
    now = datetime.utcnow()
    one_time_schedule = {'day': now.day, 'month': now.month, 'year': now.year}
    transfer_job_request = storage_transfer.CreateTransferJobRequest({'transfer_job': {'project_id': project_id, 'description': description, 'status': storage_transfer.TransferJob.Status.ENABLED, 'schedule': {'schedule_start_date': one_time_schedule, 'schedule_end_date': one_time_schedule}, 'transfer_spec': {'azure_blob_storage_data_source': {'storage_account': azure_storage_account, 'azure_credentials': {'sas_token': azure_sas_token}, 'container': source_container}, 'gcs_data_sink': {'bucket_name': sink_bucket}}}})
    result = client.create_transfer_job(transfer_job_request)
    print(f'Created transferJob: {result.name}')