from google.cloud import retail_v2beta

def sample_get_control():
    if False:
        for i in range(10):
            print('nop')
    client = retail_v2beta.ControlServiceClient()
    request = retail_v2beta.GetControlRequest(name='name_value')
    response = client.get_control(request=request)
    print(response)