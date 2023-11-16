from google.cloud import compute_v1

def sample_delete():
    if False:
        return 10
    client = compute_v1.RegionSslCertificatesClient()
    request = compute_v1.DeleteRegionSslCertificateRequest(project='project_value', region='region_value', ssl_certificate='ssl_certificate_value')
    response = client.delete(request=request)
    print(response)