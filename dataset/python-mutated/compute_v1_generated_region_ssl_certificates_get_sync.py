from google.cloud import compute_v1

def sample_get():
    if False:
        return 10
    client = compute_v1.RegionSslCertificatesClient()
    request = compute_v1.GetRegionSslCertificateRequest(project='project_value', region='region_value', ssl_certificate='ssl_certificate_value')
    response = client.get(request=request)
    print(response)