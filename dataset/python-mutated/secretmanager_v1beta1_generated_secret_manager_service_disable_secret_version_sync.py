from google.cloud import secretmanager_v1beta1

def sample_disable_secret_version():
    if False:
        while True:
            i = 10
    client = secretmanager_v1beta1.SecretManagerServiceClient()
    request = secretmanager_v1beta1.DisableSecretVersionRequest(name='name_value')
    response = client.disable_secret_version(request=request)
    print(response)