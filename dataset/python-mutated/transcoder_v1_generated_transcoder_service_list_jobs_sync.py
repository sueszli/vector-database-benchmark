from google.cloud.video import transcoder_v1

def sample_list_jobs():
    if False:
        print('Hello World!')
    client = transcoder_v1.TranscoderServiceClient()
    request = transcoder_v1.ListJobsRequest(parent='parent_value')
    page_result = client.list_jobs(request=request)
    for response in page_result:
        print(response)