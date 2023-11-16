from google.cloud import discoveryengine_v1alpha

def sample_create_engine():
    if False:
        for i in range(10):
            print('nop')
    client = discoveryengine_v1alpha.EngineServiceClient()
    engine = discoveryengine_v1alpha.Engine()
    engine.display_name = 'display_name_value'
    engine.solution_type = 'SOLUTION_TYPE_CHAT'
    request = discoveryengine_v1alpha.CreateEngineRequest(parent='parent_value', engine=engine, engine_id='engine_id_value')
    operation = client.create_engine(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)