import google.cloud.security.privateca_v1 as privateca_v1
from google.protobuf import duration_pb2

def create_subordinate_ca(project_id: str, location: str, ca_pool_name: str, subordinate_ca_name: str, common_name: str, organization: str, domain: str, ca_duration: int) -> None:
    if False:
        while True:
            i = 10
    '\n    Create Certificate Authority (CA) which is the subordinate CA in the given CA Pool.\n    Args:\n        project_id: project ID or project number of the Cloud project you want to use.\n        location: location you want to use. For a list of locations, see: https://cloud.google.com/certificate-authority-service/docs/locations.\n        ca_pool_name: set it to the CA Pool under which the CA should be created.\n        subordinate_ca_name: unique name for the Subordinate CA.\n        common_name: a title for your certificate authority.\n        organization: the name of your company for your certificate authority.\n        domain: the name of your company for your certificate authority.\n        ca_duration: the validity of the certificate authority in seconds.\n    '
    ca_service_client = privateca_v1.CertificateAuthorityServiceClient()
    key_version_spec = privateca_v1.CertificateAuthority.KeyVersionSpec(algorithm=privateca_v1.CertificateAuthority.SignHashAlgorithm.RSA_PKCS1_4096_SHA256)
    subject_config = privateca_v1.CertificateConfig.SubjectConfig(subject=privateca_v1.Subject(common_name=common_name, organization=organization), subject_alt_name=privateca_v1.SubjectAltNames(dns_names=[domain]))
    x509_parameters = privateca_v1.X509Parameters(key_usage=privateca_v1.KeyUsage(base_key_usage=privateca_v1.KeyUsage.KeyUsageOptions(crl_sign=True, cert_sign=True)), ca_options=privateca_v1.X509Parameters.CaOptions(is_ca=True))
    certificate_authority = privateca_v1.CertificateAuthority(type_=privateca_v1.CertificateAuthority.Type.SUBORDINATE, key_spec=key_version_spec, config=privateca_v1.CertificateConfig(subject_config=subject_config, x509_config=x509_parameters), lifetime=duration_pb2.Duration(seconds=ca_duration))
    ca_pool_path = ca_service_client.ca_pool_path(project_id, location, ca_pool_name)
    request = privateca_v1.CreateCertificateAuthorityRequest(parent=ca_pool_path, certificate_authority_id=subordinate_ca_name, certificate_authority=certificate_authority)
    operation = ca_service_client.create_certificate_authority(request=request)
    result = operation.result()
    print(f'Operation result: {result}')