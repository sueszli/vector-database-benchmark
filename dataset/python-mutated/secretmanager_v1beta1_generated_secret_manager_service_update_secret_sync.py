from google.cloud import secretmanager_v1beta1

def sample_update_secret():
    if False:
        return 10
    client = secretmanager_v1beta1.SecretManagerServiceClient()
    request = secretmanager_v1beta1.UpdateSecretRequest()
    response = client.update_secret(request=request)
    print(response)