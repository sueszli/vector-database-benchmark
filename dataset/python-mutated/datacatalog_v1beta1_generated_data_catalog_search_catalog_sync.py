from google.cloud import datacatalog_v1beta1

def sample_search_catalog():
    if False:
        i = 10
        return i + 15
    client = datacatalog_v1beta1.DataCatalogClient()
    request = datacatalog_v1beta1.SearchCatalogRequest()
    page_result = client.search_catalog(request=request)
    for response in page_result:
        print(response)