from google.cloud import kms_v1

def sample_asymmetric_decrypt():
    if False:
        print('Hello World!')
    client = kms_v1.KeyManagementServiceClient()
    request = kms_v1.AsymmetricDecryptRequest(name='name_value', ciphertext=b'ciphertext_blob')
    response = client.asymmetric_decrypt(request=request)
    print(response)