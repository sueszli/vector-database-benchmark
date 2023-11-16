from google.cloud import certificate_manager_v1

def sample_delete_dns_authorization():
    if False:
        while True:
            i = 10
    client = certificate_manager_v1.CertificateManagerClient()
    request = certificate_manager_v1.DeleteDnsAuthorizationRequest(name='name_value')
    operation = client.delete_dns_authorization(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)