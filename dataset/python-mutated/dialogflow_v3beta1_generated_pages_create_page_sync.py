from google.cloud import dialogflowcx_v3beta1

def sample_create_page():
    if False:
        while True:
            i = 10
    client = dialogflowcx_v3beta1.PagesClient()
    page = dialogflowcx_v3beta1.Page()
    page.display_name = 'display_name_value'
    request = dialogflowcx_v3beta1.CreatePageRequest(parent='parent_value', page=page)
    response = client.create_page(request=request)
    print(response)