import google.cloud.security.privateca_v1 as privateca_v1
from google.protobuf import field_mask_pb2
from google.type import expr_pb2

def update_ca_pool_issuance_policy(project_id: str, location: str, ca_pool_name: str) -> None:
    if False:
        while True:
            i = 10
    '\n    Update the issuance policy for a CA Pool. All certificates issued from this CA Pool should\n    meet the issuance policy\n\n    Args:\n        project_id: project ID or project number of the Cloud project you want to use.\n        location: location you want to use. For a list of locations, see: https://cloud.google.com/certificate-authority-service/docs/locations.\n        ca_pool_name: a unique name for the ca pool.\n    '
    caServiceClient = privateca_v1.CertificateAuthorityServiceClient()
    ca_pool_path = caServiceClient.ca_pool_path(project_id, location, ca_pool_name)
    expr = expr_pb2.Expr(expression='subject_alt_names.all(san, san.type == DNS && (san.value == "us.google.org" || san.value.endsWith(".google.com")) )')
    issuance_policy = privateca_v1.CaPool.IssuancePolicy(identity_constraints=privateca_v1.CertificateIdentityConstraints(allow_subject_passthrough=True, allow_subject_alt_names_passthrough=True, cel_expression=expr))
    ca_pool = privateca_v1.CaPool(name=ca_pool_path, issuance_policy=issuance_policy)
    request = privateca_v1.UpdateCaPoolRequest(ca_pool=ca_pool, update_mask=field_mask_pb2.FieldMask(paths=['issuance_policy.identity_constraints.allow_subject_alt_names_passthrough', 'issuance_policy.identity_constraints.allow_subject_passthrough', 'issuance_policy.identity_constraints.cel_expression']))
    operation = caServiceClient.update_ca_pool(request=request)
    result = operation.result()
    print('Operation result', result)
    issuance_policy = caServiceClient.get_ca_pool(name=ca_pool_path).issuance_policy
    if issuance_policy.identity_constraints.allow_subject_passthrough and issuance_policy.identity_constraints.allow_subject_alt_names_passthrough:
        print('CA Pool Issuance policy has been updated successfully!')
        return
    print('Error in updating CA Pool Issuance policy! Please try again!')