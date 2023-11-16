from google.cloud import retail_v2alpha

def sample_update_catalog():
    if False:
        i = 10
        return i + 15
    client = retail_v2alpha.CatalogServiceClient()
    catalog = retail_v2alpha.Catalog()
    catalog.name = 'name_value'
    catalog.display_name = 'display_name_value'
    request = retail_v2alpha.UpdateCatalogRequest(catalog=catalog)
    response = client.update_catalog(request=request)
    print(response)