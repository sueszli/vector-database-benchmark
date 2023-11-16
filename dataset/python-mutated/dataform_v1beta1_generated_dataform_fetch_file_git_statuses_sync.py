from google.cloud import dataform_v1beta1

def sample_fetch_file_git_statuses():
    if False:
        return 10
    client = dataform_v1beta1.DataformClient()
    request = dataform_v1beta1.FetchFileGitStatusesRequest(name='name_value')
    response = client.fetch_file_git_statuses(request=request)
    print(response)