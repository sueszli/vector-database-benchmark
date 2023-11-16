from google.cloud import kms_v1

def sample_update_ekm_connection():
    if False:
        return 10
    client = kms_v1.EkmServiceClient()
    request = kms_v1.UpdateEkmConnectionRequest()
    response = client.update_ekm_connection(request=request)
    print(response)