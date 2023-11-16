from google.cloud import secretmanager_v1

def sample_access_secret_version():
    if False:
        print('Hello World!')
    client = secretmanager_v1.SecretManagerServiceClient()
    request = secretmanager_v1.AccessSecretVersionRequest(name='name_value')
    response = client.access_secret_version(request=request)
    print(response)