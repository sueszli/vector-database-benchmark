import google.cloud.security.privateca_v1 as privateca_v1

def activate_subordinate_ca(project_id: str, location: str, ca_pool_name: str, subordinate_ca_name: str, pem_ca_certificate: str, ca_name: str) -> None:
    if False:
        for i in range(10):
            print('nop')
    "\n    Activate a subordinate Certificate Authority (CA).\n    *Prerequisite*: Get the Certificate Signing Resource (CSR) of the subordinate CA signed by another CA. Pass in the signed\n    certificate and (issuer CA's name or the issuer CA's Certificate chain).\n    *Post*: After activating the subordinate CA, it should be enabled before issuing certificates.\n    Args:\n        project_id: project ID or project number of the Cloud project you want to use.\n        location: location you want to use. For a list of locations, see: https://cloud.google.com/certificate-authority-service/docs/locations.\n        ca_pool_name: set it to the CA Pool under which the CA should be created.\n        pem_ca_certificate: the signed certificate, obtained by signing the CSR.\n        subordinate_ca_name: the CA to be activated.\n        ca_name: The name of the certificate authority which signed the CSR.\n            If an external CA (CA not present in Google Cloud) was used for signing,\n            then use the CA's issuerCertificateChain.\n    "
    ca_service_client = privateca_v1.CertificateAuthorityServiceClient()
    subordinate_ca_path = ca_service_client.certificate_authority_path(project_id, location, ca_pool_name, subordinate_ca_name)
    ca_path = ca_service_client.certificate_authority_path(project_id, location, ca_pool_name, ca_name)
    subordinate_config = privateca_v1.SubordinateConfig(certificate_authority=ca_path)
    request = privateca_v1.ActivateCertificateAuthorityRequest(name=subordinate_ca_path, pem_ca_certificate=pem_ca_certificate, subordinate_config=subordinate_config)
    operation = ca_service_client.activate_certificate_authority(request=request)
    result = operation.result()
    print('Operation result:', result)
    print(f'Current state: {ca_service_client.get_certificate_authority(name=subordinate_ca_path).state}')