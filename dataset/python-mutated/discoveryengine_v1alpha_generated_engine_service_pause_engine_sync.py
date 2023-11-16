from google.cloud import discoveryengine_v1alpha

def sample_pause_engine():
    if False:
        for i in range(10):
            print('nop')
    client = discoveryengine_v1alpha.EngineServiceClient()
    request = discoveryengine_v1alpha.PauseEngineRequest(name='name_value')
    response = client.pause_engine(request=request)
    print(response)