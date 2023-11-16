from google.cloud import appengine_admin_v1

def sample_list_authorized_certificates():
    if False:
        for i in range(10):
            print('nop')
    client = appengine_admin_v1.AuthorizedCertificatesClient()
    request = appengine_admin_v1.ListAuthorizedCertificatesRequest()
    page_result = client.list_authorized_certificates(request=request)
    for response in page_result:
        print(response)