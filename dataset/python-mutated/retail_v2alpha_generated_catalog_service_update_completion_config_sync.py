from google.cloud import retail_v2alpha

def sample_update_completion_config():
    if False:
        while True:
            i = 10
    client = retail_v2alpha.CatalogServiceClient()
    completion_config = retail_v2alpha.CompletionConfig()
    completion_config.name = 'name_value'
    request = retail_v2alpha.UpdateCompletionConfigRequest(completion_config=completion_config)
    response = client.update_completion_config(request=request)
    print(response)