from google.cloud import kms_v1

def sample_raw_encrypt():
    if False:
        print('Hello World!')
    client = kms_v1.KeyManagementServiceClient()
    request = kms_v1.RawEncryptRequest(name='name_value', plaintext=b'plaintext_blob')
    response = client.raw_encrypt(request=request)
    print(response)