from google.cloud import secretmanager_v1beta1

def sample_enable_secret_version():
    if False:
        i = 10
        return i + 15
    client = secretmanager_v1beta1.SecretManagerServiceClient()
    request = secretmanager_v1beta1.EnableSecretVersionRequest(name='name_value')
    response = client.enable_secret_version(request=request)
    print(response)