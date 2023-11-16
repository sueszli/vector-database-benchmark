from google.cloud.video import transcoder_v1

def sample_get_job():
    if False:
        for i in range(10):
            print('nop')
    client = transcoder_v1.TranscoderServiceClient()
    request = transcoder_v1.GetJobRequest(name='name_value')
    response = client.get_job(request=request)
    print(response)