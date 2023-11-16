from google.cloud import retail_v2beta

def sample_get_attributes_config():
    if False:
        print('Hello World!')
    client = retail_v2beta.CatalogServiceClient()
    request = retail_v2beta.GetAttributesConfigRequest(name='name_value')
    response = client.get_attributes_config(request=request)
    print(response)