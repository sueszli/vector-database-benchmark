from google.cloud import retail_v2

def sample_get_attributes_config():
    if False:
        while True:
            i = 10
    client = retail_v2.CatalogServiceClient()
    request = retail_v2.GetAttributesConfigRequest(name='name_value')
    response = client.get_attributes_config(request=request)
    print(response)