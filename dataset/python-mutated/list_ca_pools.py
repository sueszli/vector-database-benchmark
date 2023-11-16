import google.cloud.security.privateca_v1 as privateca_v1

def list_ca_pools(project_id: str, location: str) -> None:
    if False:
        while True:
            i = 10
    '\n    List all CA pools present in the given project and location.\n\n    Args:\n        project_id: project ID or project number of the Cloud project you want to use.\n        location: location you want to use. For a list of locations, see: https://cloud.google.com/certificate-authority-service/docs/locations.\n    '
    caServiceClient = privateca_v1.CertificateAuthorityServiceClient()
    location_path = caServiceClient.common_location_path(project_id, location)
    request = privateca_v1.ListCaPoolsRequest(parent=location_path)
    print('Available CA pools:')
    for ca_pool in caServiceClient.list_ca_pools(request=request):
        ca_pool_name = ca_pool.name
        print(caServiceClient.parse_ca_pool_path(ca_pool_name)['ca_pool'])