from google.cloud import recommendationengine_v1beta1

def sample_list_catalog_items():
    if False:
        print('Hello World!')
    client = recommendationengine_v1beta1.CatalogServiceClient()
    request = recommendationengine_v1beta1.ListCatalogItemsRequest(parent='parent_value')
    page_result = client.list_catalog_items(request=request)
    for response in page_result:
        print(response)