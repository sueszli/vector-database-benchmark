from google.cloud import recaptchaenterprise_v1

def create_mfa_assessment(project_id: str, recaptcha_site_key: str, token: str, recaptcha_action: str, hashed_account_id: str, email: str, phone_number: str) -> None:
    if False:
        i = 10
        return i + 15
    "Creates an assessment to obtain Multi-Factor Authentication result.\n\n    If the result is unspecified, sends the request token to the caller to initiate MFA challenge.\n\n    Args:\n        project_id: GCloud Project ID\n        recaptcha_site_key: Site key obtained by registering a domain/app to use recaptcha services.\n        token: The token obtained from the client on passing the recaptchaSiteKey.\n            To get the token, integrate the recaptchaSiteKey with frontend. See,\n            https://cloud.google.com/recaptcha-enterprise/docs/instrument-web-pages#frontend_integration_score\n        recaptcha_action: The action name corresponding to the token.\n        hashed_account_id: Create hashedAccountId from user identifier.\n            It's a one-way hash of the user identifier: HMAC SHA 256 + salt\n        email: Email id of the user to trigger the MFA challenge.\n        phone_number: Phone number of the user to trigger the MFA challenge. Phone number must be valid\n            and formatted according to the E.164 recommendation.\n            See: https://www.itu.int/rec/T-REC-E.164/en\n    "
    client = recaptchaenterprise_v1.RecaptchaEnterpriseServiceClient()
    event = recaptchaenterprise_v1.Event(site_key=recaptcha_site_key, token=token, hashed_account_id=hashed_account_id)
    endpoint_verification_info = recaptchaenterprise_v1.EndpointVerificationInfo(email_address=email, phone_number=phone_number)
    account_verification_info = recaptchaenterprise_v1.AccountVerificationInfo(endpoints=[endpoint_verification_info])
    assessment = recaptchaenterprise_v1.Assessment(event=event, account_verification=account_verification_info)
    project_name = f'projects/{project_id}'
    request = recaptchaenterprise_v1.CreateAssessmentRequest(assessment=assessment, parent=project_name)
    response = client.create_assessment(request)
    if not verify_response_integrity(response, recaptcha_action):
        raise RuntimeError('Failed to verify token integrity.')
    result = response.account_verification.latest_verification_result
    if result == recaptchaenterprise_v1.types.AccountVerificationInfo.Result.RESULT_UNSPECIFIED:
        print('Result unspecified. Trigger MFA challenge in the client by passing the request token.')
    print(f'MFA result: {result}')

def verify_response_integrity(response: recaptchaenterprise_v1.Assessment, recaptcha_action: str) -> bool:
    if False:
        while True:
            i = 10
    'Verifies the token and action integrity.'
    if not response.token_properties.valid:
        print(f'The CreateAssessment call failed because the token was invalid for the following reasons: {response.token_properties.invalid_reason}')
        return False
    if response.token_properties.action != recaptcha_action:
        print('The action attribute in your reCAPTCHA tag does not match the action you are expecting to score')
        return False
    return True