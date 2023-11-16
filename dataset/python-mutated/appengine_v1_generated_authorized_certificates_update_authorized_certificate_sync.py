from google.cloud import appengine_admin_v1

def sample_update_authorized_certificate():
    if False:
        while True:
            i = 10
    client = appengine_admin_v1.AuthorizedCertificatesClient()
    request = appengine_admin_v1.UpdateAuthorizedCertificateRequest()
    response = client.update_authorized_certificate(request=request)
    print(response)