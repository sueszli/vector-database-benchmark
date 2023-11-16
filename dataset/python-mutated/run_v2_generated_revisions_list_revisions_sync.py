from google.cloud import run_v2

def sample_list_revisions():
    if False:
        return 10
    client = run_v2.RevisionsClient()
    request = run_v2.ListRevisionsRequest(parent='parent_value')
    page_result = client.list_revisions(request=request)
    for response in page_result:
        print(response)