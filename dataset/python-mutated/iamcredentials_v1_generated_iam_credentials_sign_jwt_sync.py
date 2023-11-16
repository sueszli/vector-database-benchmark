from google.cloud import iam_credentials_v1

def sample_sign_jwt():
    if False:
        print('Hello World!')
    client = iam_credentials_v1.IAMCredentialsClient()
    request = iam_credentials_v1.SignJwtRequest(name='name_value', payload='payload_value')
    response = client.sign_jwt(request=request)
    print(response)