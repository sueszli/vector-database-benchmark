from google.cloud import kms_v1

def sample_get_key_ring():
    if False:
        print('Hello World!')
    client = kms_v1.KeyManagementServiceClient()
    request = kms_v1.GetKeyRingRequest(name='name_value')
    response = client.get_key_ring(request=request)
    print(response)