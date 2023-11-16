from google.cloud import dialogflowcx_v3beta1

def sample_update_page():
    if False:
        i = 10
        return i + 15
    client = dialogflowcx_v3beta1.PagesClient()
    page = dialogflowcx_v3beta1.Page()
    page.display_name = 'display_name_value'
    request = dialogflowcx_v3beta1.UpdatePageRequest(page=page)
    response = client.update_page(request=request)
    print(response)