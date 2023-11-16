from google.cloud import secretmanager_v1

def sample_get_secret_version():
    if False:
        return 10
    client = secretmanager_v1.SecretManagerServiceClient()
    request = secretmanager_v1.GetSecretVersionRequest(name='name_value')
    response = client.get_secret_version(request=request)
    print(response)