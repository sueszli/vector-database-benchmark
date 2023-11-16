import google.cloud.security.privateca_v1 as privateca_v1

def disable_certificate_authority(project_id: str, location: str, ca_pool_name: str, ca_name: str) -> None:
    if False:
        while True:
            i = 10
    '\n    Disable a Certificate Authority which is present in the given CA pool.\n\n    Args:\n        project_id: project ID or project number of the Cloud project you want to use.\n        location: location you want to use. For a list of locations, see: https://cloud.google.com/certificate-authority-service/docs/locations.\n        ca_pool_name: the name of the CA pool under which the CA is present.\n        ca_name: the name of the CA to be disabled.\n    '
    caServiceClient = privateca_v1.CertificateAuthorityServiceClient()
    ca_path = caServiceClient.certificate_authority_path(project_id, location, ca_pool_name, ca_name)
    request = privateca_v1.DisableCertificateAuthorityRequest(name=ca_path)
    operation = caServiceClient.disable_certificate_authority(request=request)
    operation.result()
    ca_state = caServiceClient.get_certificate_authority(name=ca_path).state
    if ca_state == privateca_v1.CertificateAuthority.State.DISABLED:
        print('Disabled Certificate Authority:', ca_name)
    else:
        print('Cannot disable the Certificate Authority ! Current CA State:', ca_state)