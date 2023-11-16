from google.cloud import discoveryengine_v1alpha

def sample_resume_engine():
    if False:
        print('Hello World!')
    client = discoveryengine_v1alpha.EngineServiceClient()
    request = discoveryengine_v1alpha.ResumeEngineRequest(name='name_value')
    response = client.resume_engine(request=request)
    print(response)