from google.cloud import recaptchaenterprise_v1

def sample_retrieve_legacy_secret_key():
    if False:
        return 10
    client = recaptchaenterprise_v1.RecaptchaEnterpriseServiceClient()
    request = recaptchaenterprise_v1.RetrieveLegacySecretKeyRequest(key='key_value')
    response = client.retrieve_legacy_secret_key(request=request)
    print(response)