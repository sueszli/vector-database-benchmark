from google.cloud import secretmanager_v1beta1

def sample_list_secret_versions():
    if False:
        i = 10
        return i + 15
    client = secretmanager_v1beta1.SecretManagerServiceClient()
    request = secretmanager_v1beta1.ListSecretVersionsRequest(parent='parent_value')
    page_result = client.list_secret_versions(request=request)
    for response in page_result:
        print(response)