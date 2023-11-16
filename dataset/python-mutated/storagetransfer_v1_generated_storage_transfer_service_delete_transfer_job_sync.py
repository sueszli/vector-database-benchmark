from google.cloud import storage_transfer_v1

def sample_delete_transfer_job():
    if False:
        for i in range(10):
            print('nop')
    client = storage_transfer_v1.StorageTransferServiceClient()
    request = storage_transfer_v1.DeleteTransferJobRequest(job_name='job_name_value', project_id='project_id_value')
    client.delete_transfer_job(request=request)