from google.cloud import retail_v2beta

def sample_create_serving_config():
    if False:
        i = 10
        return i + 15
    client = retail_v2beta.ServingConfigServiceClient()
    serving_config = retail_v2beta.ServingConfig()
    serving_config.display_name = 'display_name_value'
    serving_config.solution_types = ['SOLUTION_TYPE_SEARCH']
    request = retail_v2beta.CreateServingConfigRequest(parent='parent_value', serving_config=serving_config, serving_config_id='serving_config_id_value')
    response = client.create_serving_config(request=request)
    print(response)