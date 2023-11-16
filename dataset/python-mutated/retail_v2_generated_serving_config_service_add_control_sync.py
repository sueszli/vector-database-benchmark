from google.cloud import retail_v2

def sample_add_control():
    if False:
        i = 10
        return i + 15
    client = retail_v2.ServingConfigServiceClient()
    request = retail_v2.AddControlRequest(serving_config='serving_config_value', control_id='control_id_value')
    response = client.add_control(request=request)
    print(response)