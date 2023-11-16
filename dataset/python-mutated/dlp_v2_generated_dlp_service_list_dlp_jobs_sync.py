from google.cloud import dlp_v2

def sample_list_dlp_jobs():
    if False:
        for i in range(10):
            print('nop')
    client = dlp_v2.DlpServiceClient()
    request = dlp_v2.ListDlpJobsRequest(parent='parent_value')
    page_result = client.list_dlp_jobs(request=request)
    for response in page_result:
        print(response)