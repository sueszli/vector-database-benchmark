from google.cloud.devtools import cloudbuild_v2

def sample_fetch_linkable_repositories():
    if False:
        i = 10
        return i + 15
    client = cloudbuild_v2.RepositoryManagerClient()
    request = cloudbuild_v2.FetchLinkableRepositoriesRequest(connection='connection_value')
    page_result = client.fetch_linkable_repositories(request=request)
    for response in page_result:
        print(response)