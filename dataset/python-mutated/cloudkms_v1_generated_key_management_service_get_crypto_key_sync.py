from google.cloud import kms_v1

def sample_get_crypto_key():
    if False:
        print('Hello World!')
    client = kms_v1.KeyManagementServiceClient()
    request = kms_v1.GetCryptoKeyRequest(name='name_value')
    response = client.get_crypto_key(request=request)
    print(response)