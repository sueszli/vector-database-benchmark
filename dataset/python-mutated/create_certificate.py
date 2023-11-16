import google.cloud.security.privateca_v1 as privateca_v1
from google.protobuf import duration_pb2

def create_certificate(project_id: str, location: str, ca_pool_name: str, ca_name: str, certificate_name: str, common_name: str, domain_name: str, certificate_lifetime: int, public_key_bytes: bytes) -> None:
    if False:
        i = 10
        return i + 15
    '\n    Create a Certificate which is issued by the Certificate Authority present in the CA Pool.\n    The key used to sign the certificate is created by the Cloud KMS.\n\n    Args:\n        project_id: project ID or project number of the Cloud project you want to use.\n        location: location you want to use. For a list of locations, see: https://cloud.google.com/certificate-authority-service/docs/locations.\n        ca_pool_name: set a unique name for the CA pool.\n        ca_name: the name of the certificate authority which issues the certificate.\n        certificate_name: set a unique name for the certificate.\n        common_name: a title for your certificate.\n        domain_name: fully qualified domain name for your certificate.\n        certificate_lifetime: the validity of the certificate in seconds.\n        public_key_bytes: public key used in signing the certificates.\n    '
    caServiceClient = privateca_v1.CertificateAuthorityServiceClient()
    public_key = privateca_v1.PublicKey(key=public_key_bytes, format_=privateca_v1.PublicKey.KeyFormat.PEM)
    subject_config = privateca_v1.CertificateConfig.SubjectConfig(subject=privateca_v1.Subject(common_name=common_name), subject_alt_name=privateca_v1.SubjectAltNames(dns_names=[domain_name]))
    x509_parameters = privateca_v1.X509Parameters(key_usage=privateca_v1.KeyUsage(base_key_usage=privateca_v1.KeyUsage.KeyUsageOptions(digital_signature=True, key_encipherment=True), extended_key_usage=privateca_v1.KeyUsage.ExtendedKeyUsageOptions(server_auth=True, client_auth=True)))
    certificate = privateca_v1.Certificate(config=privateca_v1.CertificateConfig(public_key=public_key, subject_config=subject_config, x509_config=x509_parameters), lifetime=duration_pb2.Duration(seconds=certificate_lifetime))
    request = privateca_v1.CreateCertificateRequest(parent=caServiceClient.ca_pool_path(project_id, location, ca_pool_name), certificate_id=certificate_name, certificate=certificate, issuing_certificate_authority_id=ca_name)
    result = caServiceClient.create_certificate(request=request)
    print('Certificate creation result:', result)