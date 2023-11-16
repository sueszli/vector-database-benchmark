from google.cloud import datacatalog_v1

def sample_search_catalog():
    if False:
        return 10
    client = datacatalog_v1.DataCatalogClient()
    request = datacatalog_v1.SearchCatalogRequest()
    page_result = client.search_catalog(request=request)
    for response in page_result:
        print(response)