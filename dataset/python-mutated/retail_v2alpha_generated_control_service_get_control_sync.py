from google.cloud import retail_v2alpha

def sample_get_control():
    if False:
        while True:
            i = 10
    client = retail_v2alpha.ControlServiceClient()
    request = retail_v2alpha.GetControlRequest(name='name_value')
    response = client.get_control(request=request)
    print(response)