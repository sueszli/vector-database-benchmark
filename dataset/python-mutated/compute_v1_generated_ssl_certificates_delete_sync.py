from google.cloud import compute_v1

def sample_delete():
    if False:
        for i in range(10):
            print('nop')
    client = compute_v1.SslCertificatesClient()
    request = compute_v1.DeleteSslCertificateRequest(project='project_value', ssl_certificate='ssl_certificate_value')
    response = client.delete(request=request)
    print(response)