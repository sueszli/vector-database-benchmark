import google.cloud.security.privateca_v1 as privateca_v1

def undelete_certificate_authority(project_id: str, location: str, ca_pool_name: str, ca_name: str) -> None:
    if False:
        return 10
    '\n    Restore a deleted CA, if still within the grace period of 30 days.\n\n    Args:\n        project_id: project ID or project number of the Cloud project you want to use.\n        location: location you want to use. For a list of locations, see: https://cloud.google.com/certificate-authority-service/docs/locations.\n        ca_pool_name: the name of the CA pool under which the deleted CA is present.\n        ca_name: the name of the CA to be restored (undeleted).\n    '
    caServiceClient = privateca_v1.CertificateAuthorityServiceClient()
    ca_path = caServiceClient.certificate_authority_path(project_id, location, ca_pool_name, ca_name)
    ca_state = caServiceClient.get_certificate_authority(name=ca_path).state
    if ca_state != privateca_v1.CertificateAuthority.State.DELETED:
        print('CA is not deleted !')
        return
    request = privateca_v1.UndeleteCertificateAuthorityRequest(name=ca_path)
    operation = caServiceClient.undelete_certificate_authority(request=request)
    result = operation.result()
    print('Operation result', result)
    ca_state = caServiceClient.get_certificate_authority(name=ca_path).state
    if ca_state == privateca_v1.CertificateAuthority.State.DISABLED:
        print('Successfully undeleted Certificate Authority:', ca_name)
    else:
        print('Unable to restore the Certificate Authority! Please try again! Current state:', ca_state)