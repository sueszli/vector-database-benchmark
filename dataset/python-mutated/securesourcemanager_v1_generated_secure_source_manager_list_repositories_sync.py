from google.cloud import securesourcemanager_v1

def sample_list_repositories():
    if False:
        while True:
            i = 10
    client = securesourcemanager_v1.SecureSourceManagerClient()
    request = securesourcemanager_v1.ListRepositoriesRequest(parent='parent_value')
    page_result = client.list_repositories(request=request)
    for response in page_result:
        print(response)