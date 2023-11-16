from google.cloud import alloydb_v1alpha

def sample_generate_client_certificate():
    if False:
        i = 10
        return i + 15
    client = alloydb_v1alpha.AlloyDBAdminClient()
    request = alloydb_v1alpha.GenerateClientCertificateRequest(parent='parent_value')
    response = client.generate_client_certificate(request=request)
    print(response)