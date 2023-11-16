from google.cloud import discoveryengine_v1alpha

def sample_tune_engine():
    if False:
        for i in range(10):
            print('nop')
    client = discoveryengine_v1alpha.EngineServiceClient()
    request = discoveryengine_v1alpha.TuneEngineRequest(name='name_value')
    operation = client.tune_engine(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)