from google.cloud import retail_v2alpha

def sample_delete_control():
    if False:
        for i in range(10):
            print('nop')
    client = retail_v2alpha.ControlServiceClient()
    request = retail_v2alpha.DeleteControlRequest(name='name_value')
    client.delete_control(request=request)