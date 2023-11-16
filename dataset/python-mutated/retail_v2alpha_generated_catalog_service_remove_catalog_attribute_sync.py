from google.cloud import retail_v2alpha

def sample_remove_catalog_attribute():
    if False:
        print('Hello World!')
    client = retail_v2alpha.CatalogServiceClient()
    request = retail_v2alpha.RemoveCatalogAttributeRequest(attributes_config='attributes_config_value', key='key_value')
    response = client.remove_catalog_attribute(request=request)
    print(response)