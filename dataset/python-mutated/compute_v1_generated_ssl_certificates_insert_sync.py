from google.cloud import compute_v1

def sample_insert():
    if False:
        for i in range(10):
            print('nop')
    client = compute_v1.SslCertificatesClient()
    request = compute_v1.InsertSslCertificateRequest(project='project_value')
    response = client.insert(request=request)
    print(response)