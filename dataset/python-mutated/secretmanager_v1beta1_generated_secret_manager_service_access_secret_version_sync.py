from google.cloud import secretmanager_v1beta1

def sample_access_secret_version():
    if False:
        return 10
    client = secretmanager_v1beta1.SecretManagerServiceClient()
    request = secretmanager_v1beta1.AccessSecretVersionRequest(name='name_value')
    response = client.access_secret_version(request=request)
    print(response)