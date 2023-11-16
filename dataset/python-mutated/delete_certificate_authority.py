import google.cloud.security.privateca_v1 as privateca_v1

def delete_certificate_authority(project_id: str, location: str, ca_pool_name: str, ca_name: str) -> None:
    if False:
        i = 10
        return i + 15
    '\n    Delete the Certificate Authority from the specified CA pool.\n    Before deletion, the CA must be disabled and must not contain any active certificates.\n\n    Args:\n        project_id: project ID or project number of the Cloud project you want to use.\n        location: location you want to use. For a list of locations, see: https://cloud.google.com/certificate-authority-service/docs/locations.\n        ca_pool_name: the name of the CA pool under which the CA is present.\n        ca_name: the name of the CA to be deleted.\n    '
    caServiceClient = privateca_v1.CertificateAuthorityServiceClient()
    ca_path = caServiceClient.certificate_authority_path(project_id, location, ca_pool_name, ca_name)
    ca_state = caServiceClient.get_certificate_authority(name=ca_path).state
    if ca_state != privateca_v1.CertificateAuthority.State.DISABLED:
        print('Please disable the Certificate Authority before deletion ! Current state:', ca_state)
        raise RuntimeError(f'You can only delete disabled Certificate Authorities. {ca_name} is not disabled!')
    request = privateca_v1.DeleteCertificateAuthorityRequest(name=ca_path, ignore_active_certificates=False)
    operation = caServiceClient.delete_certificate_authority(request=request)
    result = operation.result()
    print('Operation result', result)
    ca_state = caServiceClient.get_certificate_authority(name=ca_path).state
    if ca_state == privateca_v1.CertificateAuthority.State.DELETED:
        print('Successfully deleted Certificate Authority:', ca_name)
    else:
        print('Unable to delete Certificate Authority. Please try again ! Current state:', ca_state)