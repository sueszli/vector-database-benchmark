from google.cloud.security import privateca_v1beta1

def sample_list_reusable_configs():
    if False:
        while True:
            i = 10
    client = privateca_v1beta1.CertificateAuthorityServiceClient()
    request = privateca_v1beta1.ListReusableConfigsRequest(parent='parent_value')
    page_result = client.list_reusable_configs(request=request)
    for response in page_result:
        print(response)