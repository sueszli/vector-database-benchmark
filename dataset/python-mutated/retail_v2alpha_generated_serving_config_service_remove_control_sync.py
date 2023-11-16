from google.cloud import retail_v2alpha

def sample_remove_control():
    if False:
        while True:
            i = 10
    client = retail_v2alpha.ServingConfigServiceClient()
    request = retail_v2alpha.RemoveControlRequest(serving_config='serving_config_value', control_id='control_id_value')
    response = client.remove_control(request=request)
    print(response)