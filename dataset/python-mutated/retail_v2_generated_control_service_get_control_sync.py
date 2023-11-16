from google.cloud import retail_v2

def sample_get_control():
    if False:
        for i in range(10):
            print('nop')
    client = retail_v2.ControlServiceClient()
    request = retail_v2.GetControlRequest(name='name_value')
    response = client.get_control(request=request)
    print(response)