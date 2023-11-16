from google.cloud import privatecatalog_v1beta1

def sample_search_products():
    if False:
        i = 10
        return i + 15
    client = privatecatalog_v1beta1.PrivateCatalogClient()
    request = privatecatalog_v1beta1.SearchProductsRequest(resource='resource_value')
    page_result = client.search_products(request=request)
    for response in page_result:
        print(response)