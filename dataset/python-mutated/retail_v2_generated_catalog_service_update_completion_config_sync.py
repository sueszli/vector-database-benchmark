from google.cloud import retail_v2

def sample_update_completion_config():
    if False:
        for i in range(10):
            print('nop')
    client = retail_v2.CatalogServiceClient()
    completion_config = retail_v2.CompletionConfig()
    completion_config.name = 'name_value'
    request = retail_v2.UpdateCompletionConfigRequest(completion_config=completion_config)
    response = client.update_completion_config(request=request)
    print(response)