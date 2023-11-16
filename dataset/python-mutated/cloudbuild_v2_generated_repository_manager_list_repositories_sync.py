from google.cloud.devtools import cloudbuild_v2

def sample_list_repositories():
    if False:
        for i in range(10):
            print('nop')
    client = cloudbuild_v2.RepositoryManagerClient()
    request = cloudbuild_v2.ListRepositoriesRequest(parent='parent_value')
    page_result = client.list_repositories(request=request)
    for response in page_result:
        print(response)