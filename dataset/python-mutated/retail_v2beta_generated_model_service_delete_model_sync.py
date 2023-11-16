from google.cloud import retail_v2beta

def sample_delete_model():
    if False:
        while True:
            i = 10
    client = retail_v2beta.ModelServiceClient()
    request = retail_v2beta.DeleteModelRequest(name='name_value')
    client.delete_model(request=request)