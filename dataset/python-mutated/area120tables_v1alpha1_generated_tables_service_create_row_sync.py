from google.area120 import tables_v1alpha1

def sample_create_row():
    if False:
        print('Hello World!')
    client = tables_v1alpha1.TablesServiceClient()
    request = tables_v1alpha1.CreateRowRequest(parent='parent_value')
    response = client.create_row(request=request)
    print(response)