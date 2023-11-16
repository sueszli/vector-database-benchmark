from google.cloud import retail_v2alpha

def sample_get_attributes_config():
    if False:
        return 10
    client = retail_v2alpha.CatalogServiceClient()
    request = retail_v2alpha.GetAttributesConfigRequest(name='name_value')
    response = client.get_attributes_config(request=request)
    print(response)