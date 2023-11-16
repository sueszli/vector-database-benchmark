from google.area120 import tables_v1alpha1

def sample_batch_create_rows():
    if False:
        while True:
            i = 10
    client = tables_v1alpha1.TablesServiceClient()
    requests = tables_v1alpha1.CreateRowRequest()
    requests.parent = 'parent_value'
    request = tables_v1alpha1.BatchCreateRowsRequest(parent='parent_value', requests=requests)
    response = client.batch_create_rows(request=request)
    print(response)