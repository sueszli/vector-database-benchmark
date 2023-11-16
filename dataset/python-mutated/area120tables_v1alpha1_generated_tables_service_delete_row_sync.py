from google.area120 import tables_v1alpha1

def sample_delete_row():
    if False:
        while True:
            i = 10
    client = tables_v1alpha1.TablesServiceClient()
    request = tables_v1alpha1.DeleteRowRequest(name='name_value')
    client.delete_row(request=request)