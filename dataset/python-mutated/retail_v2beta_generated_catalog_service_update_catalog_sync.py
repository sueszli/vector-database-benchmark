from google.cloud import retail_v2beta

def sample_update_catalog():
    if False:
        i = 10
        return i + 15
    client = retail_v2beta.CatalogServiceClient()
    catalog = retail_v2beta.Catalog()
    catalog.name = 'name_value'
    catalog.display_name = 'display_name_value'
    request = retail_v2beta.UpdateCatalogRequest(catalog=catalog)
    response = client.update_catalog(request=request)
    print(response)