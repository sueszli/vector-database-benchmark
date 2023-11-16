from google.cloud import dialogflowcx_v3

def sample_list_changelogs():
    if False:
        for i in range(10):
            print('nop')
    client = dialogflowcx_v3.ChangelogsClient()
    request = dialogflowcx_v3.ListChangelogsRequest(parent='parent_value')
    page_result = client.list_changelogs(request=request)
    for response in page_result:
        print(response)