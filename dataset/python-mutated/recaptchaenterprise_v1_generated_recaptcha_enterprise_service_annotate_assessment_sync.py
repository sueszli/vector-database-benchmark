from google.cloud import recaptchaenterprise_v1

def sample_annotate_assessment():
    if False:
        for i in range(10):
            print('nop')
    client = recaptchaenterprise_v1.RecaptchaEnterpriseServiceClient()
    request = recaptchaenterprise_v1.AnnotateAssessmentRequest(name='name_value')
    response = client.annotate_assessment(request=request)
    print(response)