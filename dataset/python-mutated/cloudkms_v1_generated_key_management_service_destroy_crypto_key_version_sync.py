from google.cloud import kms_v1

def sample_destroy_crypto_key_version():
    if False:
        while True:
            i = 10
    client = kms_v1.KeyManagementServiceClient()
    request = kms_v1.DestroyCryptoKeyVersionRequest(name='name_value')
    response = client.destroy_crypto_key_version(request=request)
    print(response)