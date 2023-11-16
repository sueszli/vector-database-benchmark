from google.cloud import retail_v2

def sample_update_catalog():
    if False:
        while True:
            i = 10
    client = retail_v2.CatalogServiceClient()
    catalog = retail_v2.Catalog()
    catalog.name = 'name_value'
    catalog.display_name = 'display_name_value'
    request = retail_v2.UpdateCatalogRequest(catalog=catalog)
    response = client.update_catalog(request=request)
    print(response)