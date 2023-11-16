from google.cloud import storage_transfer_v1

def sample_get_google_service_account():
    if False:
        return 10
    client = storage_transfer_v1.StorageTransferServiceClient()
    request = storage_transfer_v1.GetGoogleServiceAccountRequest(project_id='project_id_value')
    response = client.get_google_service_account(request=request)
    print(response)