import google.cloud.security.privateca_v1 as privateca_v1

def filter_certificates(project_id: str, location: str, ca_pool_name: str, filter_condition: str) -> None:
    if False:
        for i in range(10):
            print('nop')
    '\n    Filter certificates based on a condition and list them.\n\n    Args:\n        project_id: project ID or project number of the Cloud project you want to use.\n        location: location you want to use. For a list of locations, see: https://cloud.google.com/certificate-authority-service/docs/locations.\n        ca_pool_name: name of the CA pool which contains the certificates to be listed.\n    '
    caServiceClient = privateca_v1.CertificateAuthorityServiceClient()
    ca_pool_path = caServiceClient.ca_pool_path(project_id, location, ca_pool_name)
    request = privateca_v1.ListCertificatesRequest(parent=ca_pool_path, filter=filter_condition)
    print('Available certificates: ')
    for cert in caServiceClient.list_certificates(request=request):
        print(f'- {cert.name}')