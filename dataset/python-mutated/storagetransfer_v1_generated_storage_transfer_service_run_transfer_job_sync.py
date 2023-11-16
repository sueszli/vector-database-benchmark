from google.cloud import storage_transfer_v1

def sample_run_transfer_job():
    if False:
        while True:
            i = 10
    client = storage_transfer_v1.StorageTransferServiceClient()
    request = storage_transfer_v1.RunTransferJobRequest(job_name='job_name_value', project_id='project_id_value')
    operation = client.run_transfer_job(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)