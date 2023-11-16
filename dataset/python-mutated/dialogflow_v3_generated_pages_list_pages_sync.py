from google.cloud import dialogflowcx_v3

def sample_list_pages():
    if False:
        i = 10
        return i + 15
    client = dialogflowcx_v3.PagesClient()
    request = dialogflowcx_v3.ListPagesRequest(parent='parent_value')
    page_result = client.list_pages(request=request)
    for response in page_result:
        print(response)