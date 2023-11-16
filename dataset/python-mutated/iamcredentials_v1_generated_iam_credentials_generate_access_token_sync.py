from google.cloud import iam_credentials_v1

def sample_generate_access_token():
    if False:
        print('Hello World!')
    client = iam_credentials_v1.IAMCredentialsClient()
    request = iam_credentials_v1.GenerateAccessTokenRequest(name='name_value', scope=['scope_value1', 'scope_value2'])
    response = client.generate_access_token(request=request)
    print(response)