from google.cloud import appengine_admin_v1

def sample_get_authorized_certificate():
    if False:
        i = 10
        return i + 15
    client = appengine_admin_v1.AuthorizedCertificatesClient()
    request = appengine_admin_v1.GetAuthorizedCertificateRequest()
    response = client.get_authorized_certificate(request=request)
    print(response)