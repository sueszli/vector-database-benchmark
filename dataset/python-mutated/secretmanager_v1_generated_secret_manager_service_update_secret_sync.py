from google.cloud import secretmanager_v1

def sample_update_secret():
    if False:
        for i in range(10):
            print('nop')
    client = secretmanager_v1.SecretManagerServiceClient()
    request = secretmanager_v1.UpdateSecretRequest()
    response = client.update_secret(request=request)
    print(response)