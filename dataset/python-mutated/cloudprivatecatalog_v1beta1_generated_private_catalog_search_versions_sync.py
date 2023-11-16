from google.cloud import privatecatalog_v1beta1

def sample_search_versions():
    if False:
        print('Hello World!')
    client = privatecatalog_v1beta1.PrivateCatalogClient()
    request = privatecatalog_v1beta1.SearchVersionsRequest(resource='resource_value', query='query_value')
    page_result = client.search_versions(request=request)
    for response in page_result:
        print(response)