import google.cloud.security.privateca_v1 as privateca_v1
from google.protobuf import duration_pb2

def create_certificate_csr(project_id: str, location: str, ca_pool_name: str, ca_name: str, certificate_name: str, certificate_lifetime: int, pem_csr: str) -> None:
    if False:
        print('Hello World!')
    '\n    Create a Certificate which is issued by the specified Certificate Authority (CA).\n    The certificate details and the public key is provided as a Certificate Signing Request (CSR).\n    Args:\n        project_id: project ID or project number of the Cloud project you want to use.\n        location: location you want to use. For a list of locations, see: https://cloud.google.com/certificate-authority-service/docs/locations.\n        ca_pool_name: set a unique name for the CA pool.\n        ca_name: the name of the certificate authority to sign the CSR.\n        certificate_name: set a unique name for the certificate.\n        certificate_lifetime: the validity of the certificate in seconds.\n        pem_csr: set the Certificate Issuing Request in the pem encoded format.\n    '
    ca_service_client = privateca_v1.CertificateAuthorityServiceClient()
    certificate = privateca_v1.Certificate(pem_csr=pem_csr, lifetime=duration_pb2.Duration(seconds=certificate_lifetime))
    request = privateca_v1.CreateCertificateRequest(parent=ca_service_client.ca_pool_path(project_id, location, ca_pool_name), certificate_id=certificate_name, certificate=certificate, issuing_certificate_authority_id=ca_name)
    response = ca_service_client.create_certificate(request=request)
    print(f'Certificate created successfully: {response.name}')
    print(f'Signed certificate: {response.pem_certificate}')
    print(f'Issuer chain list: {response.pem_certificate_chain}')