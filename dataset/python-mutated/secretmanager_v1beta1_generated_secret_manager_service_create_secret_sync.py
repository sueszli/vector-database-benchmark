from google.cloud import secretmanager_v1beta1

def sample_create_secret():
    if False:
        i = 10
        return i + 15
    client = secretmanager_v1beta1.SecretManagerServiceClient()
    request = secretmanager_v1beta1.CreateSecretRequest(parent='parent_value', secret_id='secret_id_value')
    response = client.create_secret(request=request)
    print(response)