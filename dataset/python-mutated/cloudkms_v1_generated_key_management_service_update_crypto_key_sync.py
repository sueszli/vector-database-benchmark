from google.cloud import kms_v1

def sample_update_crypto_key():
    if False:
        while True:
            i = 10
    client = kms_v1.KeyManagementServiceClient()
    request = kms_v1.UpdateCryptoKeyRequest()
    response = client.update_crypto_key(request=request)
    print(response)