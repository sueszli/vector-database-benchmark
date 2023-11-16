from google.cloud import retail_v2beta

def sample_delete_control():
    if False:
        return 10
    client = retail_v2beta.ControlServiceClient()
    request = retail_v2beta.DeleteControlRequest(name='name_value')
    client.delete_control(request=request)