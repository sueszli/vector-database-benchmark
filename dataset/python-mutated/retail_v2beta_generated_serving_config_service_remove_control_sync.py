from google.cloud import retail_v2beta

def sample_remove_control():
    if False:
        return 10
    client = retail_v2beta.ServingConfigServiceClient()
    request = retail_v2beta.RemoveControlRequest(serving_config='serving_config_value', control_id='control_id_value')
    response = client.remove_control(request=request)
    print(response)