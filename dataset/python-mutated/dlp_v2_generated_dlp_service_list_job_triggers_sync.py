from google.cloud import dlp_v2

def sample_list_job_triggers():
    if False:
        while True:
            i = 10
    client = dlp_v2.DlpServiceClient()
    request = dlp_v2.ListJobTriggersRequest(parent='parent_value')
    page_result = client.list_job_triggers(request=request)
    for response in page_result:
        print(response)