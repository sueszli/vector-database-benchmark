from google.cloud import retail_v2

def sample_get_completion_config():
    if False:
        for i in range(10):
            print('nop')
    client = retail_v2.CatalogServiceClient()
    request = retail_v2.GetCompletionConfigRequest(name='name_value')
    response = client.get_completion_config(request=request)
    print(response)