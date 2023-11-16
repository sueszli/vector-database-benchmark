from google.cloud import retail_v2beta

def sample_add_control():
    if False:
        return 10
    client = retail_v2beta.ServingConfigServiceClient()
    request = retail_v2beta.AddControlRequest(serving_config='serving_config_value', control_id='control_id_value')
    response = client.add_control(request=request)
    print(response)