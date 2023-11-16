from google.cloud import dialogflowcx_v3

def sample_update_page():
    if False:
        for i in range(10):
            print('nop')
    client = dialogflowcx_v3.PagesClient()
    page = dialogflowcx_v3.Page()
    page.display_name = 'display_name_value'
    request = dialogflowcx_v3.UpdatePageRequest(page=page)
    response = client.update_page(request=request)
    print(response)