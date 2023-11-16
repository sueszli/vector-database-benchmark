from google.cloud import kms_v1

def sample_create_crypto_key():
    if False:
        print('Hello World!')
    client = kms_v1.KeyManagementServiceClient()
    request = kms_v1.CreateCryptoKeyRequest(parent='parent_value', crypto_key_id='crypto_key_id_value')
    response = client.create_crypto_key(request=request)
    print(response)