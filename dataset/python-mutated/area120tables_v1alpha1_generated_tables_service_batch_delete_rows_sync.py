from google.area120 import tables_v1alpha1

def sample_batch_delete_rows():
    if False:
        for i in range(10):
            print('nop')
    client = tables_v1alpha1.TablesServiceClient()
    request = tables_v1alpha1.BatchDeleteRowsRequest(parent='parent_value', names=['names_value1', 'names_value2'])
    client.batch_delete_rows(request=request)