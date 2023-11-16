from google.cloud.video import transcoder_v1

def sample_list_job_templates():
    if False:
        i = 10
        return i + 15
    client = transcoder_v1.TranscoderServiceClient()
    request = transcoder_v1.ListJobTemplatesRequest(parent='parent_value')
    page_result = client.list_job_templates(request=request)
    for response in page_result:
        print(response)