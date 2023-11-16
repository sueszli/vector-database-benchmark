from google.cloud.devtools import cloudbuild_v2

def sample_fetch_git_refs():
    if False:
        i = 10
        return i + 15
    client = cloudbuild_v2.RepositoryManagerClient()
    request = cloudbuild_v2.FetchGitRefsRequest(repository='repository_value')
    response = client.fetch_git_refs(request=request)
    print(response)