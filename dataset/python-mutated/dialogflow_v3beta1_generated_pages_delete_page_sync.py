from google.cloud import dialogflowcx_v3beta1

def sample_delete_page():
    if False:
        for i in range(10):
            print('nop')
    client = dialogflowcx_v3beta1.PagesClient()
    request = dialogflowcx_v3beta1.DeletePageRequest(name='name_value')
    client.delete_page(request=request)