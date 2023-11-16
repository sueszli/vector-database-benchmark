from google.cloud import retail_v2

def sample_remove_control():
    if False:
        for i in range(10):
            print('nop')
    client = retail_v2.ServingConfigServiceClient()
    request = retail_v2.RemoveControlRequest(serving_config='serving_config_value', control_id='control_id_value')
    response = client.remove_control(request=request)
    print(response)