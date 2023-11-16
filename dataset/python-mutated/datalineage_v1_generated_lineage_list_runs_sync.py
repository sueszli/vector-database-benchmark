from google.cloud import datacatalog_lineage_v1

def sample_list_runs():
    if False:
        for i in range(10):
            print('nop')
    client = datacatalog_lineage_v1.LineageClient()
    request = datacatalog_lineage_v1.ListRunsRequest(parent='parent_value')
    page_result = client.list_runs(request=request)
    for response in page_result:
        print(response)