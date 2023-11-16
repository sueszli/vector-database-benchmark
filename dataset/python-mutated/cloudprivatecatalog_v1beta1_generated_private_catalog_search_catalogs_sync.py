from google.cloud import privatecatalog_v1beta1

def sample_search_catalogs():
    if False:
        print('Hello World!')
    client = privatecatalog_v1beta1.PrivateCatalogClient()
    request = privatecatalog_v1beta1.SearchCatalogsRequest(resource='resource_value')
    page_result = client.search_catalogs(request=request)
    for response in page_result:
        print(response)