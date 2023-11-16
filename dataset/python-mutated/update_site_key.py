import time
from google.cloud import recaptchaenterprise_v1

def update_site_key(project_id: str, recaptcha_site_key: str, domain_name: str) -> None:
    if False:
        i = 10
        return i + 15
    '\n    Update the properties of the given site key present under the project id.\n\n    Args:\n    project_id: GCloud Project ID.\n    recaptcha_site_key: Specify the site key.\n    domain_name: Specify the domain name for which the settings should be updated.\n    '
    client = recaptchaenterprise_v1.RecaptchaEnterpriseServiceClient()
    key_name = f'projects/{project_id}/keys/{recaptcha_site_key}'
    web_settings = recaptchaenterprise_v1.WebKeySettings()
    web_settings.allow_amp_traffic = True
    web_settings.allowed_domains.append(domain_name)
    key = recaptchaenterprise_v1.Key()
    key.display_name = 'any descriptive name for the key'
    key.name = key_name
    key.web_settings = web_settings
    update_key_request = recaptchaenterprise_v1.UpdateKeyRequest()
    update_key_request.key = key
    client.update_key(update_key_request)
    time.sleep(5)
    get_key_request = recaptchaenterprise_v1.GetKeyRequest()
    get_key_request.name = key_name
    response = client.get_key(get_key_request)
    web_settings = response.web_settings
    if not web_settings.allow_amp_traffic:
        print("Error! reCAPTCHA Site key property hasn't been updated. Please try again !")
    else:
        print('reCAPTCHA Site key successfully updated ! ')
if __name__ == '__main__':
    import google.auth
    import google.auth.exceptions
    try:
        default_project_id = google.auth.default()[1]
        recaptcha_site_key = 'recaptcha_site_key'
        domain_name = 'localhost'
    except google.auth.exceptions.DefaultCredentialsError:
        print('Please use `gcloud auth application-default login` or set GOOGLE_APPLICATION_CREDENTIALS to use this script.')
    else:
        update_site_key(default_project_id, recaptcha_site_key, domain_name)