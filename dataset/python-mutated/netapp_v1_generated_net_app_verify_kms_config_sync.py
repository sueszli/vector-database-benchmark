from google.cloud import netapp_v1

def sample_verify_kms_config():
    if False:
        i = 10
        return i + 15
    client = netapp_v1.NetAppClient()
    request = netapp_v1.VerifyKmsConfigRequest(name='name_value')
    response = client.verify_kms_config(request=request)
    print(response)