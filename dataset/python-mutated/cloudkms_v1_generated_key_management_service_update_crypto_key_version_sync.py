from google.cloud import kms_v1

def sample_update_crypto_key_version():
    if False:
        print('Hello World!')
    client = kms_v1.KeyManagementServiceClient()
    request = kms_v1.UpdateCryptoKeyVersionRequest()
    response = client.update_crypto_key_version(request=request)
    print(response)