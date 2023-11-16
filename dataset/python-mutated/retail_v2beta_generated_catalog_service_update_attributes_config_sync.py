from google.cloud import retail_v2beta

def sample_update_attributes_config():
    if False:
        while True:
            i = 10
    client = retail_v2beta.CatalogServiceClient()
    attributes_config = retail_v2beta.AttributesConfig()
    attributes_config.name = 'name_value'
    request = retail_v2beta.UpdateAttributesConfigRequest(attributes_config=attributes_config)
    response = client.update_attributes_config(request=request)
    print(response)