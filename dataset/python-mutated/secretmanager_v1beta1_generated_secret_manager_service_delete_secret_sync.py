from google.cloud import secretmanager_v1beta1

def sample_delete_secret():
    if False:
        print('Hello World!')
    client = secretmanager_v1beta1.SecretManagerServiceClient()
    request = secretmanager_v1beta1.DeleteSecretRequest(name='name_value')
    client.delete_secret(request=request)