from google.cloud import retail_v2beta

def sample_get_model():
    if False:
        print('Hello World!')
    client = retail_v2beta.ModelServiceClient()
    request = retail_v2beta.GetModelRequest(name='name_value')
    response = client.get_model(request=request)
    print(response)