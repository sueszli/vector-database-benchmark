from google.cloud import storage_transfer_v1

def sample_resume_transfer_operation():
    if False:
        while True:
            i = 10
    client = storage_transfer_v1.StorageTransferServiceClient()
    request = storage_transfer_v1.ResumeTransferOperationRequest(name='name_value')
    client.resume_transfer_operation(request=request)