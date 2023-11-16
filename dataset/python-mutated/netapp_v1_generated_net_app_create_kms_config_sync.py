from google.cloud import netapp_v1

def sample_create_kms_config():
    if False:
        for i in range(10):
            print('nop')
    client = netapp_v1.NetAppClient()
    kms_config = netapp_v1.KmsConfig()
    kms_config.crypto_key_name = 'crypto_key_name_value'
    request = netapp_v1.CreateKmsConfigRequest(parent='parent_value', kms_config_id='kms_config_id_value', kms_config=kms_config)
    operation = client.create_kms_config(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)