from google.cloud import recaptchaenterprise_v1

def sample_get_key():
    if False:
        i = 10
        return i + 15
    client = recaptchaenterprise_v1.RecaptchaEnterpriseServiceClient()
    request = recaptchaenterprise_v1.GetKeyRequest(name='name_value')
    response = client.get_key(request=request)
    print(response)