from google.cloud import retail_v2

def sample_get_model():
    if False:
        while True:
            i = 10
    client = retail_v2.ModelServiceClient()
    request = retail_v2.GetModelRequest(name='name_value')
    response = client.get_model(request=request)
    print(response)