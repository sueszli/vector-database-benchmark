from google.cloud import kms_v1

def sample_decrypt():
    if False:
        print('Hello World!')
    client = kms_v1.KeyManagementServiceClient()
    request = kms_v1.DecryptRequest(name='name_value', ciphertext=b'ciphertext_blob')
    response = client.decrypt(request=request)
    print(response)