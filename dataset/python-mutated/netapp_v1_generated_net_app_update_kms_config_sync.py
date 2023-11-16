from google.cloud import netapp_v1

def sample_update_kms_config():
    if False:
        while True:
            i = 10
    client = netapp_v1.NetAppClient()
    kms_config = netapp_v1.KmsConfig()
    kms_config.crypto_key_name = 'crypto_key_name_value'
    request = netapp_v1.UpdateKmsConfigRequest(kms_config=kms_config)
    operation = client.update_kms_config(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)