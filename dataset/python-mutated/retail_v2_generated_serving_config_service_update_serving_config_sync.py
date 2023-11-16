from google.cloud import retail_v2

def sample_update_serving_config():
    if False:
        return 10
    client = retail_v2.ServingConfigServiceClient()
    serving_config = retail_v2.ServingConfig()
    serving_config.display_name = 'display_name_value'
    serving_config.solution_types = ['SOLUTION_TYPE_SEARCH']
    request = retail_v2.UpdateServingConfigRequest(serving_config=serving_config)
    response = client.update_serving_config(request=request)
    print(response)