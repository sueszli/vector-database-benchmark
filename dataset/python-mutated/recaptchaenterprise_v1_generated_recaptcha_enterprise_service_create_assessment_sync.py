from google.cloud import recaptchaenterprise_v1

def sample_create_assessment():
    if False:
        while True:
            i = 10
    client = recaptchaenterprise_v1.RecaptchaEnterpriseServiceClient()
    request = recaptchaenterprise_v1.CreateAssessmentRequest(parent='parent_value')
    response = client.create_assessment(request=request)
    print(response)