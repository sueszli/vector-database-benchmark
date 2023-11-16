import google.cloud.security.privateca_v1 as privateca_v1
from google.type import expr_pb2

def create_certificate_template(project_id: str, location: str, certificate_template_id: str) -> None:
    if False:
        return 10
    '\n    Create a Certificate template. These templates can be reused for common\n    certificate issuance scenarios.\n\n    Args:\n        project_id: project ID or project number of the Cloud project you want to use.\n        location: location you want to use. For a list of locations, see: https://cloud.google.com/certificate-authority-service/docs/locations.\n        certificate_template_id: set a unique name for the certificate template.\n    '
    caServiceClient = privateca_v1.CertificateAuthorityServiceClient()
    x509_parameters = privateca_v1.X509Parameters(key_usage=privateca_v1.KeyUsage(base_key_usage=privateca_v1.KeyUsage.KeyUsageOptions(digital_signature=True, key_encipherment=True), extended_key_usage=privateca_v1.KeyUsage.ExtendedKeyUsageOptions(server_auth=True)), ca_options=privateca_v1.X509Parameters.CaOptions(is_ca=False))
    expr = expr_pb2.Expr(expression='subject_alt_names.all(san, san.type == DNS)')
    certificate_template = privateca_v1.CertificateTemplate(predefined_values=x509_parameters, identity_constraints=privateca_v1.CertificateIdentityConstraints(cel_expression=expr, allow_subject_passthrough=False, allow_subject_alt_names_passthrough=False))
    request = privateca_v1.CreateCertificateTemplateRequest(parent=caServiceClient.common_location_path(project_id, location), certificate_template=certificate_template, certificate_template_id=certificate_template_id)
    operation = caServiceClient.create_certificate_template(request=request)
    result = operation.result()
    print('Operation result:', result)