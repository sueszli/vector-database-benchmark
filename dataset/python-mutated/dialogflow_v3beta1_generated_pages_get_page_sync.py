from google.cloud import dialogflowcx_v3beta1

def sample_get_page():
    if False:
        for i in range(10):
            print('nop')
    client = dialogflowcx_v3beta1.PagesClient()
    request = dialogflowcx_v3beta1.GetPageRequest(name='name_value')
    response = client.get_page(request=request)
    print(response)