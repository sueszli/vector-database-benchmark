from google.cloud import secretmanager_v1

def sample_destroy_secret_version():
    if False:
        for i in range(10):
            print('nop')
    client = secretmanager_v1.SecretManagerServiceClient()
    request = secretmanager_v1.DestroySecretVersionRequest(name='name_value')
    response = client.destroy_secret_version(request=request)
    print(response)