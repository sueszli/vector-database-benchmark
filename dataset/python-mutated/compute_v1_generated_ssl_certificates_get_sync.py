from google.cloud import compute_v1

def sample_get():
    if False:
        print('Hello World!')
    client = compute_v1.SslCertificatesClient()
    request = compute_v1.GetSslCertificateRequest(project='project_value', ssl_certificate='ssl_certificate_value')
    response = client.get(request=request)
    print(response)