from google.area120 import tables_v1alpha1

def sample_batch_update_rows():
    if False:
        for i in range(10):
            print('nop')
    client = tables_v1alpha1.TablesServiceClient()
    request = tables_v1alpha1.BatchUpdateRowsRequest(parent='parent_value')
    response = client.batch_update_rows(request=request)
    print(response)