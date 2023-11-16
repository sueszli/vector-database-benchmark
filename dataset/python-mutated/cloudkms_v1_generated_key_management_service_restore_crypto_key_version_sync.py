from google.cloud import kms_v1

def sample_restore_crypto_key_version():
    if False:
        while True:
            i = 10
    client = kms_v1.KeyManagementServiceClient()
    request = kms_v1.RestoreCryptoKeyVersionRequest(name='name_value')
    response = client.restore_crypto_key_version(request=request)
    print(response)