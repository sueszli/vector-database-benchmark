from google.cloud import recaptchaenterprise_v1

def delete_site_key(project_id: str, recaptcha_site_key: str) -> None:
    if False:
        return 10
    'Delete the given reCAPTCHA site key present under the project ID.\n\n    Args:\n    project_id : GCloud Project ID.\n    recaptcha_site_key: Specify the key ID to be deleted.\n    '
    client = recaptchaenterprise_v1.RecaptchaEnterpriseServiceClient()
    key_name = f'projects/{project_id}/keys/{recaptcha_site_key}'
    request = recaptchaenterprise_v1.DeleteKeyRequest()
    request.name = key_name
    client.delete_key(request)
    print('reCAPTCHA Site key deleted successfully ! ')
if __name__ == '__main__':
    import google.auth
    import google.auth.exceptions
    try:
        default_project_id = google.auth.default()[1]
        recaptcha_site_key = 'recaptcha_site_key'
    except google.auth.exceptions.DefaultCredentialsError:
        print('Please use `gcloud auth application-default login` or set GOOGLE_APPLICATION_CREDENTIALS to use this script.')
    else:
        delete_site_key(default_project_id, recaptcha_site_key)