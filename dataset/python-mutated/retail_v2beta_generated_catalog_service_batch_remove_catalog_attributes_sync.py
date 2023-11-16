from google.cloud import retail_v2beta

def sample_batch_remove_catalog_attributes():
    if False:
        return 10
    client = retail_v2beta.CatalogServiceClient()
    request = retail_v2beta.BatchRemoveCatalogAttributesRequest(attributes_config='attributes_config_value', attribute_keys=['attribute_keys_value1', 'attribute_keys_value2'])
    response = client.batch_remove_catalog_attributes(request=request)
    print(response)