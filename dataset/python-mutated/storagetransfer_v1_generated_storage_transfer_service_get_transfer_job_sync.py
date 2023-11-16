from google.cloud import storage_transfer_v1

def sample_get_transfer_job():
    if False:
        while True:
            i = 10
    client = storage_transfer_v1.StorageTransferServiceClient()
    request = storage_transfer_v1.GetTransferJobRequest(job_name='job_name_value', project_id='project_id_value')
    response = client.get_transfer_job(request=request)
    print(response)