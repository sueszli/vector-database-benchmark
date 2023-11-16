from google.cloud import recaptchaenterprise_v1

def sample_create_key():
    if False:
        i = 10
        return i + 15
    client = recaptchaenterprise_v1.RecaptchaEnterpriseServiceClient()
    key = recaptchaenterprise_v1.Key()
    key.web_settings.integration_type = 'INVISIBLE'
    key.display_name = 'display_name_value'
    request = recaptchaenterprise_v1.CreateKeyRequest(parent='parent_value', key=key)
    response = client.create_key(request=request)
    print(response)