from google.cloud.security import privateca_v1beta1

def sample_get_reusable_config():
    if False:
        i = 10
        return i + 15
    client = privateca_v1beta1.CertificateAuthorityServiceClient()
    request = privateca_v1beta1.GetReusableConfigRequest(name='name_value')
    response = client.get_reusable_config(request=request)
    print(response)