from google.cloud import secretmanager_v1

def sample_create_secret():
    if False:
        print('Hello World!')
    client = secretmanager_v1.SecretManagerServiceClient()
    request = secretmanager_v1.CreateSecretRequest(parent='parent_value', secret_id='secret_id_value')
    response = client.create_secret(request=request)
    print(response)