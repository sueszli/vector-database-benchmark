from google.cloud import retail_v2

def sample_remove_catalog_attribute():
    if False:
        while True:
            i = 10
    client = retail_v2.CatalogServiceClient()
    request = retail_v2.RemoveCatalogAttributeRequest(attributes_config='attributes_config_value', key='key_value')
    response = client.remove_catalog_attribute(request=request)
    print(response)