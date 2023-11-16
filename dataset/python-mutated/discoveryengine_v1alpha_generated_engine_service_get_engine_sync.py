from google.cloud import discoveryengine_v1alpha

def sample_get_engine():
    if False:
        return 10
    client = discoveryengine_v1alpha.EngineServiceClient()
    request = discoveryengine_v1alpha.GetEngineRequest(name='name_value')
    response = client.get_engine(request=request)
    print(response)