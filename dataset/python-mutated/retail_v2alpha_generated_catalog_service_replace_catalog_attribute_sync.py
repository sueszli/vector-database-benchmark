from google.cloud import retail_v2alpha

def sample_replace_catalog_attribute():
    if False:
        for i in range(10):
            print('nop')
    client = retail_v2alpha.CatalogServiceClient()
    catalog_attribute = retail_v2alpha.CatalogAttribute()
    catalog_attribute.key = 'key_value'
    request = retail_v2alpha.ReplaceCatalogAttributeRequest(attributes_config='attributes_config_value', catalog_attribute=catalog_attribute)
    response = client.replace_catalog_attribute(request=request)
    print(response)