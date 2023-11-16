import google.cloud.security.privateca_v1 as privateca_v1

def delete_certificate_template(project_id: str, location: str, certificate_template_id: str) -> None:
    if False:
        i = 10
        return i + 15
    '\n    Delete the certificate template present in the given project and location.\n\n    Args:\n        project_id: project ID or project number of the Cloud project you want to use.\n        location: location you want to use. For a list of locations, see: https://cloud.google.com/certificate-authority-service/docs/locations.\n        certificate_template_id: set a unique name for the certificate template.\n    '
    caServiceClient = privateca_v1.CertificateAuthorityServiceClient()
    request = privateca_v1.DeleteCertificateTemplateRequest(name=caServiceClient.certificate_template_path(project_id, location, certificate_template_id))
    operation = caServiceClient.delete_certificate_template(request=request)
    result = operation.result()
    print('Operation result', result)
    print('Deleted certificate template:', certificate_template_id)