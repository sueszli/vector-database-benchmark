from google.cloud import dataform_v1beta1

def sample_fetch_remote_branches():
    if False:
        for i in range(10):
            print('nop')
    client = dataform_v1beta1.DataformClient()
    request = dataform_v1beta1.FetchRemoteBranchesRequest(name='name_value')
    response = client.fetch_remote_branches(request=request)
    print(response)