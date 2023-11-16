from google.cloud import dataform_v1beta1

def sample_fetch_git_ahead_behind():
    if False:
        i = 10
        return i + 15
    client = dataform_v1beta1.DataformClient()
    request = dataform_v1beta1.FetchGitAheadBehindRequest(name='name_value')
    response = client.fetch_git_ahead_behind(request=request)
    print(response)