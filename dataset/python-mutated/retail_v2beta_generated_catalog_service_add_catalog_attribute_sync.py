from google.cloud import retail_v2beta

def sample_add_catalog_attribute():
    if False:
        return 10
    client = retail_v2beta.CatalogServiceClient()
    catalog_attribute = retail_v2beta.CatalogAttribute()
    catalog_attribute.key = 'key_value'
    request = retail_v2beta.AddCatalogAttributeRequest(attributes_config='attributes_config_value', catalog_attribute=catalog_attribute)
    response = client.add_catalog_attribute(request=request)
    print(response)