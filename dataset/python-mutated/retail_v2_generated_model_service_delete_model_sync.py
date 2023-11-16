from google.cloud import retail_v2

def sample_delete_model():
    if False:
        i = 10
        return i + 15
    client = retail_v2.ModelServiceClient()
    request = retail_v2.DeleteModelRequest(name='name_value')
    client.delete_model(request=request)