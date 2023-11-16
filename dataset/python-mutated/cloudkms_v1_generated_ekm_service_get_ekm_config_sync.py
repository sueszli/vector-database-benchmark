from google.cloud import kms_v1

def sample_get_ekm_config():
    if False:
        return 10
    client = kms_v1.EkmServiceClient()
    request = kms_v1.GetEkmConfigRequest(name='name_value')
    response = client.get_ekm_config(request=request)
    print(response)