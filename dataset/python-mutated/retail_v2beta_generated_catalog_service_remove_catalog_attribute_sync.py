from google.cloud import retail_v2beta

def sample_remove_catalog_attribute():
    if False:
        print('Hello World!')
    client = retail_v2beta.CatalogServiceClient()
    request = retail_v2beta.RemoveCatalogAttributeRequest(attributes_config='attributes_config_value', key='key_value')
    response = client.remove_catalog_attribute(request=request)
    print(response)