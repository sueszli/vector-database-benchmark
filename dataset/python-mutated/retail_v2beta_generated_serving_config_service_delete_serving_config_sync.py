from google.cloud import retail_v2beta

def sample_delete_serving_config():
    if False:
        return 10
    client = retail_v2beta.ServingConfigServiceClient()
    request = retail_v2beta.DeleteServingConfigRequest(name='name_value')
    client.delete_serving_config(request=request)