from google.cloud import recaptchaenterprise_v1
from google.cloud.recaptchaenterprise_v1 import Assessment

def create_assessment(project_id: str, recaptcha_site_key: str, token: str) -> Assessment:
    if False:
        print('Hello World!')
    'Create an assessment to analyze the risk of a UI action.\n    Args:\n        project_id: Google Cloud Project ID\n        recaptcha_site_key: Site key obtained by registering a domain/app to use recaptcha services.\n        token: The token obtained from the client on passing the recaptchaSiteKey.\n    Returns: Assessment response.\n    '
    client = recaptchaenterprise_v1.RecaptchaEnterpriseServiceClient()
    event = recaptchaenterprise_v1.Event()
    event.site_key = recaptcha_site_key
    event.token = token
    assessment = recaptchaenterprise_v1.Assessment()
    assessment.event = event
    project_name = f'projects/{project_id}'
    request = recaptchaenterprise_v1.CreateAssessmentRequest()
    request.assessment = assessment
    request.parent = project_name
    response = client.create_assessment(request)
    return response