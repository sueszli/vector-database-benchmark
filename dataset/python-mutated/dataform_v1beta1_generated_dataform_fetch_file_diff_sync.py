from google.cloud import dataform_v1beta1

def sample_fetch_file_diff():
    if False:
        i = 10
        return i + 15
    client = dataform_v1beta1.DataformClient()
    request = dataform_v1beta1.FetchFileDiffRequest(workspace='workspace_value', path='path_value')
    response = client.fetch_file_diff(request=request)
    print(response)