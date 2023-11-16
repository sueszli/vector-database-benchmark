from google.cloud import retail_v2

def sample_add_catalog_attribute():
    if False:
        while True:
            i = 10
    client = retail_v2.CatalogServiceClient()
    catalog_attribute = retail_v2.CatalogAttribute()
    catalog_attribute.key = 'key_value'
    request = retail_v2.AddCatalogAttributeRequest(attributes_config='attributes_config_value', catalog_attribute=catalog_attribute)
    response = client.add_catalog_attribute(request=request)
    print(response)