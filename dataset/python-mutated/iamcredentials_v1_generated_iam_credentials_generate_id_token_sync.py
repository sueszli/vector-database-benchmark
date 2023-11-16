from google.cloud import iam_credentials_v1

def sample_generate_id_token():
    if False:
        print('Hello World!')
    client = iam_credentials_v1.IAMCredentialsClient()
    request = iam_credentials_v1.GenerateIdTokenRequest(name='name_value', audience='audience_value')
    response = client.generate_id_token(request=request)
    print(response)