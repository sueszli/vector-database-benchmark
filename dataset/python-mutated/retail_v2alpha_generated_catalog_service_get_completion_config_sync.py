from google.cloud import retail_v2alpha

def sample_get_completion_config():
    if False:
        i = 10
        return i + 15
    client = retail_v2alpha.CatalogServiceClient()
    request = retail_v2alpha.GetCompletionConfigRequest(name='name_value')
    response = client.get_completion_config(request=request)
    print(response)