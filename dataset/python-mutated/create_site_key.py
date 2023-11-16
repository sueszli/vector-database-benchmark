from google.cloud import recaptchaenterprise_v1

def create_site_key(project_id: str, domain_name: str) -> str:
    if False:
        while True:
            i = 10
    'Create reCAPTCHA Site key which binds a domain name to a unique key.\n    Args:\n    project_id : GCloud Project ID.\n    domain_name: Specify the domain name in which the reCAPTCHA should be activated.\n    '
    client = recaptchaenterprise_v1.RecaptchaEnterpriseServiceClient()
    web_settings = recaptchaenterprise_v1.WebKeySettings()
    web_settings.allowed_domains.append(domain_name)
    web_settings.allow_amp_traffic = False
    web_settings.integration_type = web_settings.IntegrationType.SCORE
    key = recaptchaenterprise_v1.Key()
    key.display_name = 'any descriptive name for the key'
    key.web_settings = web_settings
    request = recaptchaenterprise_v1.CreateKeyRequest()
    request.parent = f'projects/{project_id}'
    request.key = key
    response = client.create_key(request)
    recaptcha_site_key = response.name.rsplit('/', maxsplit=1)[1]
    print('reCAPTCHA Site key created successfully. Site Key: ' + recaptcha_site_key)
    return recaptcha_site_key
if __name__ == '__main__':
    import google.auth
    import google.auth.exceptions
    try:
        default_project_id = google.auth.default()[1]
        domain_name = 'localhost'
    except google.auth.exceptions.DefaultCredentialsError:
        print('Please use `gcloud auth application-default login` or set GOOGLE_APPLICATION_CREDENTIALS to use this script.')
    else:
        create_site_key(default_project_id, domain_name)