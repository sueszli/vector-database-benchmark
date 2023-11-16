from google.cloud import retail_v2

def sample_delete_control():
    if False:
        while True:
            i = 10
    client = retail_v2.ControlServiceClient()
    request = retail_v2.DeleteControlRequest(name='name_value')
    client.delete_control(request=request)