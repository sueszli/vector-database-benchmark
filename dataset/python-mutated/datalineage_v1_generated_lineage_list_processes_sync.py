from google.cloud import datacatalog_lineage_v1

def sample_list_processes():
    if False:
        i = 10
        return i + 15
    client = datacatalog_lineage_v1.LineageClient()
    request = datacatalog_lineage_v1.ListProcessesRequest(parent='parent_value')
    page_result = client.list_processes(request=request)
    for response in page_result:
        print(response)