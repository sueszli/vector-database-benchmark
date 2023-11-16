from google.cloud import retail_v2alpha

def sample_batch_remove_catalog_attributes():
    if False:
        for i in range(10):
            print('nop')
    client = retail_v2alpha.CatalogServiceClient()
    request = retail_v2alpha.BatchRemoveCatalogAttributesRequest(attributes_config='attributes_config_value', attribute_keys=['attribute_keys_value1', 'attribute_keys_value2'])
    response = client.batch_remove_catalog_attributes(request=request)
    print(response)