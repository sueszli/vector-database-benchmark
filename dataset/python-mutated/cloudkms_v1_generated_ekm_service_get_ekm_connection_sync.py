from google.cloud import kms_v1

def sample_get_ekm_connection():
    if False:
        for i in range(10):
            print('nop')
    client = kms_v1.EkmServiceClient()
    request = kms_v1.GetEkmConnectionRequest(name='name_value')
    response = client.get_ekm_connection(request=request)
    print(response)