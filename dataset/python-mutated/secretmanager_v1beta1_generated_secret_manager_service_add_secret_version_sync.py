from google.cloud import secretmanager_v1beta1

def sample_add_secret_version():
    if False:
        while True:
            i = 10
    client = secretmanager_v1beta1.SecretManagerServiceClient()
    request = secretmanager_v1beta1.AddSecretVersionRequest(parent='parent_value')
    response = client.add_secret_version(request=request)
    print(response)