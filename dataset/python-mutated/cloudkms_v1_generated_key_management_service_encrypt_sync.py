from google.cloud import kms_v1

def sample_encrypt():
    if False:
        while True:
            i = 10
    client = kms_v1.KeyManagementServiceClient()
    request = kms_v1.EncryptRequest(name='name_value', plaintext=b'plaintext_blob')
    response = client.encrypt(request=request)
    print(response)