from google.cloud import kms_v1

def sample_update_ekm_config():
    if False:
        print('Hello World!')
    client = kms_v1.EkmServiceClient()
    request = kms_v1.UpdateEkmConfigRequest()
    response = client.update_ekm_config(request=request)
    print(response)