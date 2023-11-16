from google.ai import generativelanguage_v1beta3

def sample_get_tuned_model():
    if False:
        print('Hello World!')
    client = generativelanguage_v1beta3.ModelServiceClient()
    request = generativelanguage_v1beta3.GetTunedModelRequest(name='name_value')
    response = client.get_tuned_model(request=request)
    print(response)