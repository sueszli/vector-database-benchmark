from google.cloud import config_v1

def sample_list_revisions():
    if False:
        for i in range(10):
            print('nop')
    client = config_v1.ConfigClient()
    request = config_v1.ListRevisionsRequest(parent='parent_value')
    page_result = client.list_revisions(request=request)
    for response in page_result:
        print(response)