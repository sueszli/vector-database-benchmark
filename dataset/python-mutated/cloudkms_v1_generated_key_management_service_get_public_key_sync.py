from google.cloud import kms_v1

def sample_get_public_key():
    if False:
        print('Hello World!')
    client = kms_v1.KeyManagementServiceClient()
    request = kms_v1.GetPublicKeyRequest(name='name_value')
    response = client.get_public_key(request=request)
    print(response)