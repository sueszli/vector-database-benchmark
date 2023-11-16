from google.cloud import retail_v2

def sample_update_attributes_config():
    if False:
        while True:
            i = 10
    client = retail_v2.CatalogServiceClient()
    attributes_config = retail_v2.AttributesConfig()
    attributes_config.name = 'name_value'
    request = retail_v2.UpdateAttributesConfigRequest(attributes_config=attributes_config)
    response = client.update_attributes_config(request=request)
    print(response)