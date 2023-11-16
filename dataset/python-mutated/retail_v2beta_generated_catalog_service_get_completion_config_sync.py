from google.cloud import retail_v2beta

def sample_get_completion_config():
    if False:
        return 10
    client = retail_v2beta.CatalogServiceClient()
    request = retail_v2beta.GetCompletionConfigRequest(name='name_value')
    response = client.get_completion_config(request=request)
    print(response)