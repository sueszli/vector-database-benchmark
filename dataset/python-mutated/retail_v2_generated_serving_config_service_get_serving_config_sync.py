from google.cloud import retail_v2

def sample_get_serving_config():
    if False:
        while True:
            i = 10
    client = retail_v2.ServingConfigServiceClient()
    request = retail_v2.GetServingConfigRequest(name='name_value')
    response = client.get_serving_config(request=request)
    print(response)