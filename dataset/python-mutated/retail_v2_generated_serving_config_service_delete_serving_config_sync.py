from google.cloud import retail_v2

def sample_delete_serving_config():
    if False:
        print('Hello World!')
    client = retail_v2.ServingConfigServiceClient()
    request = retail_v2.DeleteServingConfigRequest(name='name_value')
    client.delete_serving_config(request=request)