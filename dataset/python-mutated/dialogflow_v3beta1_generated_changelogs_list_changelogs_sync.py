from google.cloud import dialogflowcx_v3beta1

def sample_list_changelogs():
    if False:
        print('Hello World!')
    client = dialogflowcx_v3beta1.ChangelogsClient()
    request = dialogflowcx_v3beta1.ListChangelogsRequest(parent='parent_value')
    page_result = client.list_changelogs(request=request)
    for response in page_result:
        print(response)