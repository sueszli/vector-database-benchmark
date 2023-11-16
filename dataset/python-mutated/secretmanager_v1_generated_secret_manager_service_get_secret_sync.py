from google.cloud import secretmanager_v1

def sample_get_secret():
    if False:
        for i in range(10):
            print('nop')
    client = secretmanager_v1.SecretManagerServiceClient()
    request = secretmanager_v1.GetSecretRequest(name='name_value')
    response = client.get_secret(request=request)
    print(response)