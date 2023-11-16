from google.cloud import netapp_v1

def sample_get_kms_config():
    if False:
        return 10
    client = netapp_v1.NetAppClient()
    request = netapp_v1.GetKmsConfigRequest(name='name_value')
    response = client.get_kms_config(request=request)
    print(response)