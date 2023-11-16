from google.area120 import tables_v1alpha1

def sample_get_table():
    if False:
        i = 10
        return i + 15
    client = tables_v1alpha1.TablesServiceClient()
    request = tables_v1alpha1.GetTableRequest(name='name_value')
    response = client.get_table(request=request)
    print(response)