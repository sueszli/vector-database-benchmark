from google.cloud import kms_v1

def sample_raw_decrypt():
    if False:
        for i in range(10):
            print('nop')
    client = kms_v1.KeyManagementServiceClient()
    request = kms_v1.RawDecryptRequest(name='name_value', ciphertext=b'ciphertext_blob', initialization_vector=b'initialization_vector_blob')
    response = client.raw_decrypt(request=request)
    print(response)