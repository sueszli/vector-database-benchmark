from google.cloud import discoveryengine_v1alpha

def sample_delete_engine():
    if False:
        for i in range(10):
            print('nop')
    client = discoveryengine_v1alpha.EngineServiceClient()
    request = discoveryengine_v1alpha.DeleteEngineRequest(name='name_value')
    operation = client.delete_engine(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)