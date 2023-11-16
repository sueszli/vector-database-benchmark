from google.cloud.video import transcoder_v1

def sample_create_job_template():
    if False:
        i = 10
        return i + 15
    client = transcoder_v1.TranscoderServiceClient()
    request = transcoder_v1.CreateJobTemplateRequest(parent='parent_value', job_template_id='job_template_id_value')
    response = client.create_job_template(request=request)
    print(response)