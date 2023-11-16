import google.cloud.security.privateca_v1 as privateca_v1
from google.protobuf import field_mask_pb2

def update_certificate_template(project_id: str, location: str, certificate_template_id: str) -> None:
    if False:
        return 10
    '\n    Update an existing certificate template.\n\n    Args:\n        project_id: project ID or project number of the Cloud project you want to use.\n        location: location you want to use. For a list of locations, see: https://cloud.google.com/certificate-authority-service/docs/locations.\n        certificate_template_id: set a unique name for the certificate template.\n    '
    caServiceClient = privateca_v1.CertificateAuthorityServiceClient()
    certificate_name = caServiceClient.certificate_template_path(project_id, location, certificate_template_id)
    certificate_template = privateca_v1.CertificateTemplate(name=certificate_name, identity_constraints=privateca_v1.CertificateIdentityConstraints(allow_subject_passthrough=False, allow_subject_alt_names_passthrough=True))
    field_mask = field_mask_pb2.FieldMask(paths=['identity_constraints.allow_subject_alt_names_passthrough', 'identity_constraints.allow_subject_passthrough'])
    request = privateca_v1.UpdateCertificateTemplateRequest(certificate_template=certificate_template, update_mask=field_mask)
    operation = caServiceClient.update_certificate_template(request=request)
    result = operation.result()
    print('Operation result', result)
    cert_identity_constraints = caServiceClient.get_certificate_template(name=certificate_name).identity_constraints
    if not cert_identity_constraints.allow_subject_passthrough and cert_identity_constraints.allow_subject_alt_names_passthrough:
        print('Successfully updated the certificate template!')
        return
    print('Error in updating certificate template!')