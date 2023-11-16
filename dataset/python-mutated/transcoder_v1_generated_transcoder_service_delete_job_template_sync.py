from google.cloud.video import transcoder_v1

def sample_delete_job_template():
    if False:
        for i in range(10):
            print('nop')
    client = transcoder_v1.TranscoderServiceClient()
    request = transcoder_v1.DeleteJobTemplateRequest(name='name_value')
    client.delete_job_template(request=request)