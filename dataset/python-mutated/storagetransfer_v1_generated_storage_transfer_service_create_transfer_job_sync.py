from google.cloud import storage_transfer_v1

def sample_create_transfer_job():
    if False:
        return 10
    client = storage_transfer_v1.StorageTransferServiceClient()
    request = storage_transfer_v1.CreateTransferJobRequest()
    response = client.create_transfer_job(request=request)
    print(response)