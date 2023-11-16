from google.cloud import dataplex_v1

def sample_list_jobs():
    if False:
        i = 10
        return i + 15
    client = dataplex_v1.DataplexServiceClient()
    request = dataplex_v1.ListJobsRequest(parent='parent_value')
    page_result = client.list_jobs(request=request)
    for response in page_result:
        print(response)