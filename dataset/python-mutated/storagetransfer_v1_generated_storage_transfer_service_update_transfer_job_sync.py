from google.cloud import storage_transfer_v1

def sample_update_transfer_job():
    if False:
        return 10
    client = storage_transfer_v1.StorageTransferServiceClient()
    request = storage_transfer_v1.UpdateTransferJobRequest(job_name='job_name_value', project_id='project_id_value')
    response = client.update_transfer_job(request=request)
    print(response)