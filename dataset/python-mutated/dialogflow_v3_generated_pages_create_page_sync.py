from google.cloud import dialogflowcx_v3

def sample_create_page():
    if False:
        return 10
    client = dialogflowcx_v3.PagesClient()
    page = dialogflowcx_v3.Page()
    page.display_name = 'display_name_value'
    request = dialogflowcx_v3.CreatePageRequest(parent='parent_value', page=page)
    response = client.create_page(request=request)
    print(response)