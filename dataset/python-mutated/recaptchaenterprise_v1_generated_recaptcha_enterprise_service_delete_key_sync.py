from google.cloud import recaptchaenterprise_v1

def sample_delete_key():
    if False:
        print('Hello World!')
    client = recaptchaenterprise_v1.RecaptchaEnterpriseServiceClient()
    request = recaptchaenterprise_v1.DeleteKeyRequest(name='name_value')
    client.delete_key(request=request)