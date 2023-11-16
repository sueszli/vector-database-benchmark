import google.cloud.security.privateca_v1 as privateca_v1

def list_certificates(project_id: str, location: str, ca_pool_name: str) -> None:
    if False:
        i = 10
        return i + 15
    '\n    List Certificates present in the given CA pool.\n\n    Args:\n        project_id: project ID or project number of the Cloud project you want to use.\n        location: location you want to use. For a list of locations, see: https://cloud.google.com/certificate-authority-service/docs/locations.\n        ca_pool_name: name of the CA pool which contains the certificates to be listed.\n    '
    caServiceClient = privateca_v1.CertificateAuthorityServiceClient()
    ca_pool_path = caServiceClient.ca_pool_path(project_id, location, ca_pool_name)
    print(f'Available certificates in CA pool {ca_pool_name}:')
    for certificate in caServiceClient.list_certificates(parent=ca_pool_path):
        print(certificate.name)