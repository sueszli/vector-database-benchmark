from google.cloud import recaptchaenterprise_v1
from list_site_keys import list_site_keys

def migrate_site_key(project_id: str, recaptcha_site_key: str) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Migrate a key from reCAPTCHA (non-Enterprise) to reCAPTCHA Enterprise.\n        If you created the key using Admin console: https://www.google.com/recaptcha/admin/site,\n        then use this API to migrate to reCAPTCHA Enterprise.\n        For more info, see: https://cloud.google.com/recaptcha-enterprise/docs/migrate-recaptcha\n    Args:\n    project_id: Google Cloud Project ID.\n    recaptcha_site_key: Specify the site key to migrate.\n    '
    client = recaptchaenterprise_v1.RecaptchaEnterpriseServiceClient()
    name = f'projects/{project_id}/keys/{recaptcha_site_key}'
    request = recaptchaenterprise_v1.MigrateKeyRequest()
    request.name = name
    response = client.migrate_key(request)
    for key in list_site_keys(project_id):
        if key.name == response.name:
            print(f'Key migrated successfully: {recaptcha_site_key}')