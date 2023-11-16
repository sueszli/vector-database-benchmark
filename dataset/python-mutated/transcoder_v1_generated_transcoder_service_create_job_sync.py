from google.cloud.video import transcoder_v1

def sample_create_job():
    if False:
        for i in range(10):
            print('nop')
    client = transcoder_v1.TranscoderServiceClient()
    job = transcoder_v1.Job()
    job.template_id = 'template_id_value'
    request = transcoder_v1.CreateJobRequest(parent='parent_value', job=job)
    response = client.create_job(request=request)
    print(response)