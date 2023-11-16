from google.cloud import secretmanager_v1beta1

def sample_get_secret():
    if False:
        print('Hello World!')
    client = secretmanager_v1beta1.SecretManagerServiceClient()
    request = secretmanager_v1beta1.GetSecretRequest(name='name_value')
    response = client.get_secret(request=request)
    print(response)