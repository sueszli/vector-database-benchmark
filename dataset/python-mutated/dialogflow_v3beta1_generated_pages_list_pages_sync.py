from google.cloud import dialogflowcx_v3beta1

def sample_list_pages():
    if False:
        return 10
    client = dialogflowcx_v3beta1.PagesClient()
    request = dialogflowcx_v3beta1.ListPagesRequest(parent='parent_value')
    page_result = client.list_pages(request=request)
    for response in page_result:
        print(response)