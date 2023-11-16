from google.cloud.video import transcoder_v1

def sample_delete_job():
    if False:
        i = 10
        return i + 15
    client = transcoder_v1.TranscoderServiceClient()
    request = transcoder_v1.DeleteJobRequest(name='name_value')
    client.delete_job(request=request)