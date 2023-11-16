from google.cloud import datacatalog_lineage_v1

def sample_list_lineage_events():
    if False:
        return 10
    client = datacatalog_lineage_v1.LineageClient()
    request = datacatalog_lineage_v1.ListLineageEventsRequest(parent='parent_value')
    page_result = client.list_lineage_events(request=request)
    for response in page_result:
        print(response)