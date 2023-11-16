from google.cloud import iam_credentials_v1

def sample_sign_blob():
    if False:
        i = 10
        return i + 15
    client = iam_credentials_v1.IAMCredentialsClient()
    request = iam_credentials_v1.SignBlobRequest(name='name_value', payload=b'payload_blob')
    response = client.sign_blob(request=request)
    print(response)