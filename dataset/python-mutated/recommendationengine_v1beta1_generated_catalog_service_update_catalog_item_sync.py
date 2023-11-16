from google.cloud import recommendationengine_v1beta1

def sample_update_catalog_item():
    if False:
        for i in range(10):
            print('nop')
    client = recommendationengine_v1beta1.CatalogServiceClient()
    catalog_item = recommendationengine_v1beta1.CatalogItem()
    catalog_item.id = 'id_value'
    catalog_item.category_hierarchies.categories = ['categories_value1', 'categories_value2']
    catalog_item.title = 'title_value'
    request = recommendationengine_v1beta1.UpdateCatalogItemRequest(name='name_value', catalog_item=catalog_item)
    response = client.update_catalog_item(request=request)
    print(response)