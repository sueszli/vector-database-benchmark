import google.cloud.security.privateca_v1 as privateca_v1

def list_certificate_templates(project_id: str, location: str) -> None:
    if False:
        return 10
    '\n    List the certificate templates present in the given project and location.\n\n    Args:\n        project_id: project ID or project number of the Cloud project you want to use.\n        location: location you want to use. For a list of locations, see: https://cloud.google.com/certificate-authority-service/docs/locations.\n    '
    caServiceClient = privateca_v1.CertificateAuthorityServiceClient()
    request = privateca_v1.ListCertificateTemplatesRequest(parent=caServiceClient.common_location_path(project_id, location))
    print('Available certificate templates:')
    for certificate_template in caServiceClient.list_certificate_templates(request=request):
        print(certificate_template.name)