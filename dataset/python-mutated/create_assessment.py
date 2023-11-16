from google.cloud import recaptchaenterprise_v1
from google.cloud.recaptchaenterprise_v1 import Assessment

def create_assessment(project_id: str, recaptcha_site_key: str, token: str, recaptcha_action: str) -> Assessment:
    if False:
        print('Hello World!')
    'Create an assessment to analyze the risk of a UI action.\n    Args:\n        project_id: GCloud Project ID\n        recaptcha_site_key: Site key obtained by registering a domain/app to use recaptcha services.\n        token: The token obtained from the client on passing the recaptchaSiteKey.\n        recaptcha_action: Action name corresponding to the token.\n    '
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
    if not response.token_properties.valid:
        print('The CreateAssessment call failed because the token was ' + 'invalid for for the following reasons: ' + str(response.token_properties.invalid_reason))
        return
    if response.token_properties.action != recaptcha_action:
        print('The action attribute in your reCAPTCHA tag does' + 'not match the action you are expecting to score')
        return
    else:
        for reason in response.risk_analysis.reasons:
            print(reason)
        print('The reCAPTCHA score for this token is: ' + str(response.risk_analysis.score))
        assessment_name = client.parse_assessment_path(response.name).get('assessment')
        print(f'Assessment name: {assessment_name}')
    return response