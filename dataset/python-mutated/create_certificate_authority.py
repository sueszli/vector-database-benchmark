import google.cloud.security.privateca_v1 as privateca_v1
from google.protobuf import duration_pb2

def create_certificate_authority(project_id: str, location: str, ca_pool_name: str, ca_name: str, common_name: str, organization: str, ca_duration: int) -> None:
    if False:
        for i in range(10):
            print('nop')
    '\n    Create Certificate Authority which is the root CA in the given CA Pool. This CA will be\n    responsible for signing certificates within this pool.\n\n    Args:\n        project_id: project ID or project number of the Cloud project you want to use.\n        location: location you want to use. For a list of locations, see: https://cloud.google.com/certificate-authority-service/docs/locations.\n        ca_pool_name: set it to the CA Pool under which the CA should be created.\n        ca_name: unique name for the CA.\n        common_name: a title for your certificate authority.\n        organization: the name of your company for your certificate authority.\n        ca_duration: the validity of the certificate authority in seconds.\n    '
    caServiceClient = privateca_v1.CertificateAuthorityServiceClient()
    key_version_spec = privateca_v1.CertificateAuthority.KeyVersionSpec(algorithm=privateca_v1.CertificateAuthority.SignHashAlgorithm.RSA_PKCS1_4096_SHA256)
    subject_config = privateca_v1.CertificateConfig.SubjectConfig(subject=privateca_v1.Subject(common_name=common_name, organization=organization))
    x509_parameters = privateca_v1.X509Parameters(key_usage=privateca_v1.KeyUsage(base_key_usage=privateca_v1.KeyUsage.KeyUsageOptions(crl_sign=True, cert_sign=True)), ca_options=privateca_v1.X509Parameters.CaOptions(is_ca=True))
    certificate_authority = privateca_v1.CertificateAuthority(type_=privateca_v1.CertificateAuthority.Type.SELF_SIGNED, key_spec=key_version_spec, config=privateca_v1.CertificateConfig(subject_config=subject_config, x509_config=x509_parameters), lifetime=duration_pb2.Duration(seconds=ca_duration))
    ca_pool_path = caServiceClient.ca_pool_path(project_id, location, ca_pool_name)
    request = privateca_v1.CreateCertificateAuthorityRequest(parent=ca_pool_path, certificate_authority_id=ca_name, certificate_authority=certificate_authority)
    operation = caServiceClient.create_certificate_authority(request=request)
    result = operation.result()
    print('Operation result:', result)