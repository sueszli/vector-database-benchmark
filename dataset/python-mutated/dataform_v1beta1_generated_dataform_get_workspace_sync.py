from google.cloud import dataform_v1beta1

def sample_get_workspace():
    if False:
        while True:
            i = 10
    client = dataform_v1beta1.DataformClient()
    request = dataform_v1beta1.GetWorkspaceRequest(name='name_value')
    response = client.get_workspace(request=request)
    print(response)