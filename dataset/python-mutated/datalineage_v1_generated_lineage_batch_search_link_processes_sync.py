from google.cloud import datacatalog_lineage_v1

def sample_batch_search_link_processes():
    if False:
        for i in range(10):
            print('nop')
    client = datacatalog_lineage_v1.LineageClient()
    request = datacatalog_lineage_v1.BatchSearchLinkProcessesRequest(parent='parent_value', links=['links_value1', 'links_value2'])
    page_result = client.batch_search_link_processes(request=request)
    for response in page_result:
        print(response)