from google.cloud import secretmanager_v1

def sample_add_secret_version():
    if False:
        print('Hello World!')
    client = secretmanager_v1.SecretManagerServiceClient()
    request = secretmanager_v1.AddSecretVersionRequest(parent='parent_value')
    response = client.add_secret_version(request=request)
    print(response)