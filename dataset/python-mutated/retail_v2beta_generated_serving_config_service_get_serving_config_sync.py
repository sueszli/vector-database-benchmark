from google.cloud import retail_v2beta

def sample_get_serving_config():
    if False:
        i = 10
        return i + 15
    client = retail_v2beta.ServingConfigServiceClient()
    request = retail_v2beta.GetServingConfigRequest(name='name_value')
    response = client.get_serving_config(request=request)
    print(response)