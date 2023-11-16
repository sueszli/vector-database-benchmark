from google.cloud import recaptchaenterprise_v1

def sample_update_key():
    if False:
        print('Hello World!')
    client = recaptchaenterprise_v1.RecaptchaEnterpriseServiceClient()
    key = recaptchaenterprise_v1.Key()
    key.web_settings.integration_type = 'INVISIBLE'
    key.display_name = 'display_name_value'
    request = recaptchaenterprise_v1.UpdateKeyRequest(key=key)
    response = client.update_key(request=request)
    print(response)