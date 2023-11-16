from google.cloud import retail_v2alpha

def sample_pause_model():
    if False:
        i = 10
        return i + 15
    client = retail_v2alpha.ModelServiceClient()
    request = retail_v2alpha.PauseModelRequest(name='name_value')
    response = client.pause_model(request=request)
    print(response)