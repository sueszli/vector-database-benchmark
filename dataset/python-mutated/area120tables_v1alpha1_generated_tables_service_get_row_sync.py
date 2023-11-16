from google.area120 import tables_v1alpha1

def sample_get_row():
    if False:
        for i in range(10):
            print('nop')
    client = tables_v1alpha1.TablesServiceClient()
    request = tables_v1alpha1.GetRowRequest(name='name_value')
    response = client.get_row(request=request)
    print(response)