from google.cloud import retail_v2

def sample_resume_model():
    if False:
        for i in range(10):
            print('nop')
    client = retail_v2.ModelServiceClient()
    request = retail_v2.ResumeModelRequest(name='name_value')
    response = client.resume_model(request=request)
    print(response)