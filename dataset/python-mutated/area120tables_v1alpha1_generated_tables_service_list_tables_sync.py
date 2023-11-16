from google.area120 import tables_v1alpha1

def sample_list_tables():
    if False:
        return 10
    client = tables_v1alpha1.TablesServiceClient()
    request = tables_v1alpha1.ListTablesRequest()
    page_result = client.list_tables(request=request)
    for response in page_result:
        print(response)