import google.cloud.security.privateca_v1 as privateca_v1

def delete_ca_pool(project_id: str, location: str, ca_pool_name: str) -> None:
    if False:
        i = 10
        return i + 15
    '\n    Delete the CA pool as mentioned by the ca_pool_name.\n    Before deleting the pool, all CAs in the pool MUST BE deleted.\n\n    Args:\n        project_id: project ID or project number of the Cloud project you want to use.\n        location: location you want to use. For a list of locations, see: https://cloud.google.com/certificate-authority-service/docs/locations.\n        ca_pool_name: the name of the CA pool to be deleted.\n    '
    caServiceClient = privateca_v1.CertificateAuthorityServiceClient()
    ca_pool_path = caServiceClient.ca_pool_path(project_id, location, ca_pool_name)
    request = privateca_v1.DeleteCaPoolRequest(name=ca_pool_path)
    caServiceClient.delete_ca_pool(request=request)
    print('Deleted CA Pool:', ca_pool_name)