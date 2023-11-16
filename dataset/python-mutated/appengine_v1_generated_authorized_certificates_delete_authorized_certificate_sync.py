from google.cloud import appengine_admin_v1

def sample_delete_authorized_certificate():
    if False:
        i = 10
        return i + 15
    client = appengine_admin_v1.AuthorizedCertificatesClient()
    request = appengine_admin_v1.DeleteAuthorizedCertificateRequest()
    client.delete_authorized_certificate(request=request)