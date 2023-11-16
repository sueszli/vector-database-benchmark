from google.cloud import retail_v2alpha

def sample_delete_model():
    if False:
        return 10
    client = retail_v2alpha.ModelServiceClient()
    request = retail_v2alpha.DeleteModelRequest(name='name_value')
    client.delete_model(request=request)