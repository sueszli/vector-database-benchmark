from google.cloud import kms_v1

def sample_update_crypto_key_primary_version():
    if False:
        while True:
            i = 10
    client = kms_v1.KeyManagementServiceClient()
    request = kms_v1.UpdateCryptoKeyPrimaryVersionRequest(name='name_value', crypto_key_version_id='crypto_key_version_id_value')
    response = client.update_crypto_key_primary_version(request=request)
    print(response)