from google.cloud import dialogflowcx_v3

def sample_get_page():
    if False:
        print('Hello World!')
    client = dialogflowcx_v3.PagesClient()
    request = dialogflowcx_v3.GetPageRequest(name='name_value')
    response = client.get_page(request=request)
    print(response)