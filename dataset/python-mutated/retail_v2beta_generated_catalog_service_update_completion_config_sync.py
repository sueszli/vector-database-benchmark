from google.cloud import retail_v2beta

def sample_update_completion_config():
    if False:
        print('Hello World!')
    client = retail_v2beta.CatalogServiceClient()
    completion_config = retail_v2beta.CompletionConfig()
    completion_config.name = 'name_value'
    request = retail_v2beta.UpdateCompletionConfigRequest(completion_config=completion_config)
    response = client.update_completion_config(request=request)
    print(response)