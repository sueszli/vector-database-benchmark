import google.cloud.security.privateca_v1 as privateca_v1

def list_certificate_authorities(project_id: str, location: str, ca_pool_name: str) -> None:
    if False:
        print('Hello World!')
    '\n    List all Certificate authorities present in the given CA Pool.\n\n    Args:\n        project_id: project ID or project number of the Cloud project you want to use.\n        location: location you want to use. For a list of locations, see: https://cloud.google.com/certificate-authority-service/docs/locations.\n        ca_pool_name: the name of the CA pool under which the CAs to be listed are present.\n    '
    caServiceClient = privateca_v1.CertificateAuthorityServiceClient()
    ca_pool_path = caServiceClient.ca_pool_path(project_id, location, ca_pool_name)
    for ca in caServiceClient.list_certificate_authorities(parent=ca_pool_path):
        print(ca.name, 'is', ca.state)