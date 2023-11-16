from google.cloud import appengine_admin_v1

def sample_list_authorized_domains():
    if False:
        for i in range(10):
            print('nop')
    client = appengine_admin_v1.AuthorizedDomainsClient()
    request = appengine_admin_v1.ListAuthorizedDomainsRequest()
    page_result = client.list_authorized_domains(request=request)
    for response in page_result:
        print(response)