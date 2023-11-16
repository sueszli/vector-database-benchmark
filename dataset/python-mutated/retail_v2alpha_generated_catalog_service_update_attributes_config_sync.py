from google.cloud import retail_v2alpha

def sample_update_attributes_config():
    if False:
        print('Hello World!')
    client = retail_v2alpha.CatalogServiceClient()
    attributes_config = retail_v2alpha.AttributesConfig()
    attributes_config.name = 'name_value'
    request = retail_v2alpha.UpdateAttributesConfigRequest(attributes_config=attributes_config)
    response = client.update_attributes_config(request=request)
    print(response)