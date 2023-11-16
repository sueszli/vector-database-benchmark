from google.area120 import tables_v1alpha1

def sample_update_row():
    if False:
        i = 10
        return i + 15
    client = tables_v1alpha1.TablesServiceClient()
    request = tables_v1alpha1.UpdateRowRequest()
    response = client.update_row(request=request)
    print(response)