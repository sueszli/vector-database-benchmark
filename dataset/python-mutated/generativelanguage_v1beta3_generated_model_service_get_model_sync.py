from google.ai import generativelanguage_v1beta3

def sample_get_model():
    if False:
        while True:
            i = 10
    client = generativelanguage_v1beta3.ModelServiceClient()
    request = generativelanguage_v1beta3.GetModelRequest(name='name_value')
    response = client.get_model(request=request)
    print(response)