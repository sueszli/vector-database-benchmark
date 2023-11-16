import google.cloud.security.privateca_v1 as privateca_v1

def create_ca_pool(project_id: str, location: str, ca_pool_name: str) -> None:
    if False:
        for i in range(10):
            print('nop')
    '\n    Create a Certificate Authority pool. All certificates created under this CA pool will\n    follow the same issuance policy, IAM policies,etc.,\n\n    Args:\n        project_id: project ID or project number of the Cloud project you want to use.\n        location: location you want to use. For a list of locations, see: https://cloud.google.com/certificate-authority-service/docs/locations.\n        ca_pool_name: a unique name for the ca pool.\n    '
    caServiceClient = privateca_v1.CertificateAuthorityServiceClient()
    ca_pool = privateca_v1.CaPool(tier=privateca_v1.CaPool.Tier.ENTERPRISE)
    location_path = caServiceClient.common_location_path(project_id, location)
    request = privateca_v1.CreateCaPoolRequest(parent=location_path, ca_pool_id=ca_pool_name, ca_pool=ca_pool)
    operation = caServiceClient.create_ca_pool(request=request)
    print('Operation result:', operation.result())