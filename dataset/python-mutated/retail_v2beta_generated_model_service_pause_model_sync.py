from google.cloud import retail_v2beta

def sample_pause_model():
    if False:
        return 10
    client = retail_v2beta.ModelServiceClient()
    request = retail_v2beta.PauseModelRequest(name='name_value')
    response = client.pause_model(request=request)
    print(response)