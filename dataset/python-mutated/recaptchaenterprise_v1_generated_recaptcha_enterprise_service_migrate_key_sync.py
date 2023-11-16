from google.cloud import recaptchaenterprise_v1

def sample_migrate_key():
    if False:
        i = 10
        return i + 15
    client = recaptchaenterprise_v1.RecaptchaEnterpriseServiceClient()
    request = recaptchaenterprise_v1.MigrateKeyRequest(name='name_value')
    response = client.migrate_key(request=request)
    print(response)