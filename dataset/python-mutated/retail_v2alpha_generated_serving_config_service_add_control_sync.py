from google.cloud import retail_v2alpha

def sample_add_control():
    if False:
        for i in range(10):
            print('nop')
    client = retail_v2alpha.ServingConfigServiceClient()
    request = retail_v2alpha.AddControlRequest(serving_config='serving_config_value', control_id='control_id_value')
    response = client.add_control(request=request)
    print(response)