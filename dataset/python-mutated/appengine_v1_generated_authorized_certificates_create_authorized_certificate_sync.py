from google.cloud import appengine_admin_v1

def sample_create_authorized_certificate():
    if False:
        for i in range(10):
            print('nop')
    client = appengine_admin_v1.AuthorizedCertificatesClient()
    request = appengine_admin_v1.CreateAuthorizedCertificateRequest()
    response = client.create_authorized_certificate(request=request)
    print(response)