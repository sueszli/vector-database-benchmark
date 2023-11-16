import sys
import google.cloud.security.privateca_v1 as privateca_v1

def revoke_certificate(project_id: str, location: str, ca_pool_name: str, certificate_name: str) -> None:
    if False:
        for i in range(10):
            print('nop')
    '\n    Revoke an issued certificate. Once revoked, the certificate will become invalid and will expire post its lifetime.\n\n    Args:\n        project_id: project ID or project number of the Cloud project you want to use.\n        location: location you want to use. For a list of locations, see: https://cloud.google.com/certificate-authority-service/docs/locations.\n        ca_pool_name: name for the CA pool which contains the certificate.\n        certificate_name: name of the certificate to be revoked.\n    '
    caServiceClient = privateca_v1.CertificateAuthorityServiceClient()
    certificate_path = caServiceClient.certificate_path(project_id, location, ca_pool_name, certificate_name)
    request = privateca_v1.RevokeCertificateRequest(name=certificate_path, reason=privateca_v1.RevocationReason.PRIVILEGE_WITHDRAWN)
    result = caServiceClient.revoke_certificate(request=request)
    print('Certificate revoke result:', result)
if __name__ == '__main__':
    revoke_certificate(project_id=sys.argv[1], location=sys.argv[2], ca_pool_name=sys.argv[3], certificate_name=sys.argv[4])