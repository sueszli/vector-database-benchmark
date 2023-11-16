from google.cloud import datacatalog_lineage_v1

def sample_search_links():
    if False:
        for i in range(10):
            print('nop')
    client = datacatalog_lineage_v1.LineageClient()
    source = datacatalog_lineage_v1.EntityReference()
    source.fully_qualified_name = 'fully_qualified_name_value'
    request = datacatalog_lineage_v1.SearchLinksRequest(source=source, parent='parent_value')
    page_result = client.search_links(request=request)
    for response in page_result:
        print(response)