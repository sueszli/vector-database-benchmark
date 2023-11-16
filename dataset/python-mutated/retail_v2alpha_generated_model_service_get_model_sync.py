from google.cloud import retail_v2alpha

def sample_get_model():
    if False:
        return 10
    client = retail_v2alpha.ModelServiceClient()
    request = retail_v2alpha.GetModelRequest(name='name_value')
    response = client.get_model(request=request)
    print(response)