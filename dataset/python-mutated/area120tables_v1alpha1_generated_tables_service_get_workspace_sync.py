from google.area120 import tables_v1alpha1

def sample_get_workspace():
    if False:
        for i in range(10):
            print('nop')
    client = tables_v1alpha1.TablesServiceClient()
    request = tables_v1alpha1.GetWorkspaceRequest(name='name_value')
    response = client.get_workspace(request=request)
    print(response)