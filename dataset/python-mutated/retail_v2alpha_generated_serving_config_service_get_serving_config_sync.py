from google.cloud import retail_v2alpha

def sample_get_serving_config():
    if False:
        i = 10
        return i + 15
    client = retail_v2alpha.ServingConfigServiceClient()
    request = retail_v2alpha.GetServingConfigRequest(name='name_value')
    response = client.get_serving_config(request=request)
    print(response)