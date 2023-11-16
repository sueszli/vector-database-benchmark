from google.cloud import secretmanager_v1beta1

def sample_get_secret_version():
    if False:
        print('Hello World!')
    client = secretmanager_v1beta1.SecretManagerServiceClient()
    request = secretmanager_v1beta1.GetSecretVersionRequest(name='name_value')
    response = client.get_secret_version(request=request)
    print(response)