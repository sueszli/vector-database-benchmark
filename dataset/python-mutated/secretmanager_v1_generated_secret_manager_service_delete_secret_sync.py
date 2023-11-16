from google.cloud import secretmanager_v1

def sample_delete_secret():
    if False:
        for i in range(10):
            print('nop')
    client = secretmanager_v1.SecretManagerServiceClient()
    request = secretmanager_v1.DeleteSecretRequest(name='name_value')
    client.delete_secret(request=request)