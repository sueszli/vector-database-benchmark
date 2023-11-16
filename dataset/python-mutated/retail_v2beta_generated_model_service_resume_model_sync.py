from google.cloud import retail_v2beta

def sample_resume_model():
    if False:
        print('Hello World!')
    client = retail_v2beta.ModelServiceClient()
    request = retail_v2beta.ResumeModelRequest(name='name_value')
    response = client.resume_model(request=request)
    print(response)