from google.cloud import storage_transfer_v1

def sample_pause_transfer_operation():
    if False:
        i = 10
        return i + 15
    client = storage_transfer_v1.StorageTransferServiceClient()
    request = storage_transfer_v1.PauseTransferOperationRequest(name='name_value')
    client.pause_transfer_operation(request=request)