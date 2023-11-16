from google.cloud import kms_v1

def sample_create_crypto_key_version():
    if False:
        while True:
            i = 10
    client = kms_v1.KeyManagementServiceClient()
    request = kms_v1.CreateCryptoKeyVersionRequest(parent='parent_value')
    response = client.create_crypto_key_version(request=request)
    print(response)