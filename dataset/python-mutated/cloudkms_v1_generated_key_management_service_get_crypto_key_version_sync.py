from google.cloud import kms_v1

def sample_get_crypto_key_version():
    if False:
        i = 10
        return i + 15
    client = kms_v1.KeyManagementServiceClient()
    request = kms_v1.GetCryptoKeyVersionRequest(name='name_value')
    response = client.get_crypto_key_version(request=request)
    print(response)