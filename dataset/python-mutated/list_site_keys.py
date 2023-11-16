from google.cloud import recaptchaenterprise_v1
from google.cloud.recaptchaenterprise_v1.services.recaptcha_enterprise_service.pagers import ListKeysPager

def list_site_keys(project_id: str) -> ListKeysPager:
    if False:
        for i in range(10):
            print('nop')
    'List all keys present under the given project ID.\n\n    Args:\n    project_id: GCloud Project ID.\n    '
    project_name = f'projects/{project_id}'
    client = recaptchaenterprise_v1.RecaptchaEnterpriseServiceClient()
    request = recaptchaenterprise_v1.ListKeysRequest()
    request.parent = project_name
    response = client.list_keys(request)
    print('Listing reCAPTCHA site keys: ')
    for (i, key) in enumerate(response):
        print(f'{str(i)}. {key.name}')
    return response
if __name__ == '__main__':
    import google.auth
    import google.auth.exceptions
    try:
        default_project_id = google.auth.default()[1]
    except google.auth.exceptions.DefaultCredentialsError:
        print('Please use `gcloud auth application-default login` or set GOOGLE_APPLICATION_CREDENTIALS to use this script.')
    else:
        list_site_keys(default_project_id)