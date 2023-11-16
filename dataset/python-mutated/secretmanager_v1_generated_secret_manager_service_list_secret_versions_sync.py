from google.cloud import secretmanager_v1

def sample_list_secret_versions():
    if False:
        while True:
            i = 10
    client = secretmanager_v1.SecretManagerServiceClient()
    request = secretmanager_v1.ListSecretVersionsRequest(parent='parent_value')
    page_result = client.list_secret_versions(request=request)
    for response in page_result:
        print(response)