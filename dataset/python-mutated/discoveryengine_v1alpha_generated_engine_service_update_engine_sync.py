from google.cloud import discoveryengine_v1alpha

def sample_update_engine():
    if False:
        return 10
    client = discoveryengine_v1alpha.EngineServiceClient()
    engine = discoveryengine_v1alpha.Engine()
    engine.display_name = 'display_name_value'
    engine.solution_type = 'SOLUTION_TYPE_CHAT'
    request = discoveryengine_v1alpha.UpdateEngineRequest(engine=engine)
    response = client.update_engine(request=request)
    print(response)