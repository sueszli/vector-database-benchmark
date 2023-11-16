from google.cloud import dialogflowcx_v3

def sample_delete_page():
    if False:
        print('Hello World!')
    client = dialogflowcx_v3.PagesClient()
    request = dialogflowcx_v3.DeletePageRequest(name='name_value')
    client.delete_page(request=request)