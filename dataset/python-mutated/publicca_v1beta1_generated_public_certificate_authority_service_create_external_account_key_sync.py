from google.cloud.security import publicca_v1beta1

def sample_create_external_account_key():
    if False:
        return 10
    client = publicca_v1beta1.PublicCertificateAuthorityServiceClient()
    request = publicca_v1beta1.CreateExternalAccountKeyRequest(parent='parent_value')
    response = client.create_external_account_key(request=request)
    print(response)