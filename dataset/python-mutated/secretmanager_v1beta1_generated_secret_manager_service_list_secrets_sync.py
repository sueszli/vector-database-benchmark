from google.cloud import secretmanager_v1beta1

def sample_list_secrets():
    if False:
        while True:
            i = 10
    client = secretmanager_v1beta1.SecretManagerServiceClient()
    request = secretmanager_v1beta1.ListSecretsRequest(parent='parent_value')
    page_result = client.list_secrets(request=request)
    for response in page_result:
        print(response)