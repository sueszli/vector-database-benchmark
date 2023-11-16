from google.cloud import alloydb_v1beta

def sample_generate_client_certificate():
    if False:
        return 10
    client = alloydb_v1beta.AlloyDBAdminClient()
    request = alloydb_v1beta.GenerateClientCertificateRequest(parent='parent_value')
    response = client.generate_client_certificate(request=request)
    print(response)