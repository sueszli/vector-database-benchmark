from google.cloud import recommendationengine_v1beta1

def sample_import_catalog_items():
    if False:
        i = 10
        return i + 15
    client = recommendationengine_v1beta1.CatalogServiceClient()
    request = recommendationengine_v1beta1.ImportCatalogItemsRequest(parent='parent_value')
    operation = client.import_catalog_items(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)