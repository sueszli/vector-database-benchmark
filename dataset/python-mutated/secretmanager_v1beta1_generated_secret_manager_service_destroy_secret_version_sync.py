from google.cloud import secretmanager_v1beta1

def sample_destroy_secret_version():
    if False:
        i = 10
        return i + 15
    client = secretmanager_v1beta1.SecretManagerServiceClient()
    request = secretmanager_v1beta1.DestroySecretVersionRequest(name='name_value')
    response = client.destroy_secret_version(request=request)
    print(response)