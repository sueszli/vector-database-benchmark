from google.cloud import recommendationengine_v1beta1

def sample_delete_catalog_item():
    if False:
        i = 10
        return i + 15
    client = recommendationengine_v1beta1.CatalogServiceClient()
    request = recommendationengine_v1beta1.DeleteCatalogItemRequest(name='name_value')
    client.delete_catalog_item(request=request)